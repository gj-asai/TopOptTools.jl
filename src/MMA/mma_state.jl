mutable struct MMAState{T<:Real}
    it::Int
    x::DesignVector{T,T}
    xprev1::DesignVector{T,T}
    xprev2::DesignVector{T,T}

    cur_obj::T
    cur_dobj::Vector{T}
    cur_cons::Vector{T}
    cur_dcons::Matrix{T}

    obj_hist::Vector{T}
end
MMAState(x0, cur_obj, cur_dobj, cur_cons, cur_dcons) = MMAState(0, x0, similar(x0), similar(x0), cur_obj, cur_dobj, cur_cons, cur_dcons, Vector{typeof(cur_obj)}())

function relative_change(state::MMAState; window::Int=1)
    window = min(length(state.obj_hist) - 1, window)

    old_mean = mean(state.obj_hist[end-window:end-1])
    new_mean = mean(state.obj_hist[end-window+1:end])
    return abs(new_mean - old_mean) / old_mean
end
