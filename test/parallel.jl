using Logging
logger = SimpleLogger(Logging.Warn)
global_logger(logger)

using MPI
MPI.Init()
comm = MPI.COMM_WORLD

using ThreadPinning
pinthreads(:affinitymask)

using Printf
include("mmsomp.jl")

# each process will work with one angle
const SCRATCH = ENV["SCRATCH"]
const rank = MPI.Comm_rank(comm)
const angles = [-90, -60, -30, 0, 30, 60, 90]
const angle = angles[rank+1]

const w_list = 0.1:0.1:0.9
const volfrac = 0.4
const rρ = 4.0
const rθ = 4.0

compliances = Float64[]
impacts = Float64[]
for wimpact in w_list
    dir = @sprintf "results/w%02d/%d/" 10 * wimpact angle
    dir = SCRATCH * replace(dir, "-" => "n")
    mkpath(dir)

    filename = dir * "mbb"

    history = mbb_minimpact_mmsomp(volfrac, rρ, rθ, wimpact; angle, filename)
    push!(compliances, history[:final_compliance])
    push!(impacts, history[:impact][end])
end

MPI.Barrier(comm)
all_compliances = MPI.Gather(compliances, comm)
all_impacts = MPI.Gather(impacts, comm)
all_w = MPI.Gather(collect(w_list), comm)
all_angles = MPI.Gather(fill(angle, length(w_list)), comm)
if rank == 0
    @show all_compliances
    @show all_impacts
    @show all_w
    @show all_angles
end
