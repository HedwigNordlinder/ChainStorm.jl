using Flowfusion: resolveprediction, mask, step
using LogExpFunctions: logsumexp

# ─── Struct-based predictor with snapshotable self-conditioning state ─────────

mutable struct TreePredictor
    model           # ChainStorm model
    chainids        # batch chainids
    resinds         # batch resinds
    batch_dim::Int  # batch size
    f               # current self-conditioning frames (mutable)
    prev_trans      # previous translation for smoothing (mutable)
    smooth::Float64 # smoothing factor
    d               # device transfer function
end

function TreePredictor(X0, b, model; d = identity, smooth = 0.6)
    batch_dim = size(tensor(X0[1]), 4)
    f, _ = model(d(zeros(Float32, 1, batch_dim)), d(X0), d(b.chainids), d(b.resinds))
    prev_trans = values(translation(f))
    TreePredictor(model, b.chainids, b.resinds, batch_dim, f, prev_trans, smooth, d)
end

function (pred::TreePredictor)(t, Xt)
    f, aalogits = pred.model(
        pred.d(t .+ zeros(Float32, 1, pred.batch_dim)),
        pred.d(Xt),
        pred.d(pred.chainids),
        pred.d(pred.resinds),
        sc_frames = pred.f
    )
    T = eltype(values(translation(f)))
    values(translation(f)) .= pred.prev_trans .* T(pred.smooth) .+ values(translation(f)) .* T(1 - pred.smooth)
    pred.f = f
    pred.prev_trans = values(translation(f))
    return cpu(values(translation(f))), ManifoldState(rotM, eachslice(cpu(values(linear(f))), dims=(3,4))), cpu(softmax(aalogits))
end

snapshot(pred::TreePredictor) = (deepcopy(pred.f), deepcopy(pred.prev_trans))

function restore!(pred::TreePredictor, snap)
    pred.f = deepcopy(snap[1])
    pred.prev_trans = deepcopy(snap[2])
    return pred
end

# ─── Tree node ────────────────────────────────────────────────────────────────

mutable struct DTSNode
    state::Tuple          # (Xlocs, Xrots, Xaas) at this node
    t_index::Int          # index into branching_points (1-based; depth in tree)
    sc_snapshot           # self-conditioning snapshot at this node
    value::Float64        # soft value estimate v̂
    visit_count::Int      # N(x)
    reward::Union{Nothing, Float64}  # terminal reward (leaf only)
    children::Vector{DTSNode}
    parent::Union{Nothing, DTSNode}
end

function DTSNode(state, t_index, sc_snapshot; parent = nothing)
    DTSNode(state, t_index, sc_snapshot, 0.0, 0, nothing, DTSNode[], parent)
end

is_terminal(node::DTSNode, n_levels::Int) = node.t_index > n_levels
is_fully_expanded(node::DTSNode, max_children::Int) = length(node.children) >= max_children

# ─── Step schedule partitioning ───────────────────────────────────────────────

function partition_steps(stps::AbstractVector, branching_points::AbstractVector)
    # Returns n_branching_points + 1 segments.
    # Segment i covers branching_points[i] to branching_points[i+1] (or 1.0 for last).
    boundaries = vcat(branching_points, [1.0])
    segments = Vector{Vector{Float32}}()
    for i in 1:length(boundaries)-1
        lo, hi = boundaries[i], boundaries[i+1]
        # Include steps in [lo, hi]. The first step of each segment is lo itself
        # (needed as s₁ in the step pair), so we include the boundary.
        seg = Float32[s for s in stps if s >= lo && s <= hi]
        if isempty(seg) || seg[1] > lo
            pushfirst!(seg, Float32(lo))
        end
        if seg[end] < hi
            push!(seg, Float32(hi))
        end
        # Deduplicate and sort
        seg = sort(unique(seg))
        push!(segments, seg)
    end
    return segments
end

# ─── Run a segment of gen steps ──────────────────────────────────────────────

function run_segment(P, state, predictor::TreePredictor, segment_steps::AbstractVector)
    Xₜ = state
    for (s₁, s₂) in zip(segment_steps, segment_steps[begin+1:end])
        hat = resolveprediction(predictor(s₁, Xₜ), Xₜ)
        Xₜ = mask(step(P, Xₜ, hat, s₁, s₂), state)
    end
    return Xₜ
end

# ─── Rollout: run from a node to terminal ────────────────────────────────────

function rollout(P, node::DTSNode, predictor::TreePredictor, segments, n_levels)
    restore!(predictor, node.sc_snapshot)
    state = node.state
    for seg_idx in node.t_index:n_levels
        state = run_segment(P, state, predictor, segments[seg_idx])
    end
    return state
end

# ─── UCT selection ────────────────────────────────────────────────────────────

function uct_score(child::DTSNode, parent_visits::Int, c_uct::Float64)
    if child.visit_count == 0
        return Inf
    end
    return child.value + c_uct * sqrt(log(parent_visits) / child.visit_count)
end

function select(node::DTSNode, max_children::Int, c_uct::Float64, n_levels::Int)
    # Walk down the tree picking highest-UCT children until we find
    # a node that can be expanded or is terminal.
    while !is_terminal(node, n_levels)
        if !is_fully_expanded(node, max_children)
            return node  # expand here
        end
        # All children exist — pick best UCT
        node = argmax(c -> uct_score(c, node.visit_count, c_uct), node.children)
    end
    return node  # terminal
end

# ─── Expansion ────────────────────────────────────────────────────────────────

function expand!(P, node::DTSNode, predictor::TreePredictor, segments, n_levels)
    if is_terminal(node, n_levels)
        return node
    end
    # Restore predictor state to this node's snapshot
    restore!(predictor, node.sc_snapshot)
    # Run the segment from this node's branching point to the next
    child_state = run_segment(P, node.state, predictor, segments[node.t_index])
    child_snap = snapshot(predictor)
    child = DTSNode(child_state, node.t_index + 1, child_snap; parent = node)
    push!(node.children, child)
    return child
end

# ─── Soft Bellman backup ─────────────────────────────────────────────────────

function backup!(node::DTSNode, reward::Float64, lambda::Float64)
    # Set reward on the leaf
    node.reward = reward
    node.value = reward
    node.visit_count += 1
    # Walk up to root
    current = node.parent
    while current !== nothing
        current.visit_count += 1
        # Soft Bellman: v̂ = (1/λ) * log( Σ exp(λ * v̂(child)) )
        if lambda == 0.0
            current.value = maximum(c.value for c in current.children)
        else
            lse = logsumexp(lambda .* [c.value for c in current.children])
            current.value = lse / lambda
        end
        current = current.parent
    end
end

# ─── Best leaf extraction ────────────────────────────────────────────────────

function best_leaf(node::DTSNode)
    if isempty(node.children)
        return node
    end
    return best_leaf(argmax(c -> c.value, node.children))
end

# ─── Main entry point ────────────────────────────────────────────────────────

function flow_treegen(
    b, model;
    reward,
    branching_points = [0.0, 0.05, 0.15, 0.4, 0.7],
    max_children = 4,
    n_iterations = 100,
    c_uct = 1.0,
    lambda = 1.0,
    steps = :default,
    smooth = 0.6,
    d = identity
)
    # Build step schedule (same as flow_quickgen)
    stps = vcat(zeros(5), S.([0.0:0.00255:0.9975;]), [0.999, 0.9998, 1.0])
    if steps isa Number
        stps = 0f0:1f0/steps:1f0
    elseif steps isa AbstractVector
        stps = steps
    end

    n_levels = length(branching_points)
    segments = partition_steps(Float32.(stps), branching_points)

    # Initialize state and predictor
    X0 = zero_state(b)
    predictor = TreePredictor(X0, b, model; d = d, smooth = smooth)
    root_snap = snapshot(predictor)

    # Create root node
    root = DTSNode(X0, 1, root_snap)

    # DTS* main loop
    for iter in 1:n_iterations
        print("*")
        # Selection
        node = select(root, max_children, c_uct, n_levels)

        # Expansion (if not terminal)
        if !is_terminal(node, n_levels) && !is_fully_expanded(node, max_children)
            node = expand!(P, node, predictor, segments, n_levels)
        end

        # Rollout to terminal
        terminal_state = rollout(P, node, predictor, segments, n_levels)

        # Evaluate reward
        r = Float64(reward(terminal_state))

        # Backup
        backup!(node, r, lambda)
    end

    println()

    # Return the best terminal trajectory
    leaf = best_leaf(root)

    # Re-run the best path to terminal to get the final state
    restore!(predictor, leaf.sc_snapshot)
    if is_terminal(leaf, n_levels)
        return leaf.state
    else
        return rollout(P, leaf, predictor, segments, n_levels)
    end
end
