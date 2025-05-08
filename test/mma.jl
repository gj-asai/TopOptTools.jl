# Numerical examples from
# Svanberg, K. (1987), The method of moving asymptotes—a new method for structural optimization. Int. J. Numer. Meth. Engng., 24: 359-373. https://doi.org/10.1002/nme.1620240207

@testset "Cantilever beam" begin
    x0 = DesignVariables(1)
    foreach(1:5) do _
        push!(x0, 5.0, 0.0, 10.0)
    end

    C1 = 0.0624
    C2 = 1.0

    f(x) = C1 * sum(x)
    df(_) = [C1, C1, C1, C1, C1]
    objective = MMA.Objective(f, df)

    g(x) = 61 / x[1]^3 + 37 / x[2]^3 + 19 / x[3]^3 + 7 / x[4]^3 + 1 / x[5]^3 - C2
    dg(x) = [-183 / x[1]^4, -111 / x[2]^4, -57 / x[3]^4, -21 / x[4]^4, -3 / x[5]^4]
    constraint = MMA.Constraints(g, dg)

    prob = MMA.MMAProblem(x0, objective, constraint)
    for _ = 1:100
        MMA.iterate(prob)
        MMA.relative_change(prob.state) < 1e-8 && break
    end
    @test prob.state.x ≈ [6.016, 5.309, 4.494, 3.502, 2.153] atol = 1e-3
    @test prob.state.cur_obj ≈ 1.340 atol = 1e-3
end

@testset "2 bar truss" begin
    x0 = DesignVariables(1)
    push!(x0, 1.5, 0.2, 4.0)
    push!(x0, 0.5, 0.1, 1.6)

    C1 = 1.0
    C2 = 0.124

    f(x) = C1 * x[1] * sqrt(1 + x[2]^2)
    df(x) = [C1 * sqrt(1 + x[2]^2), C1 * x[1] * x[2] / sqrt(1 + x[2]^2)]
    objective = MMA.Objective(f, df)

    g1(x) = C2 * sqrt(1 + x[2]^2) * (8 / x[1] + 1 / (x[1] * x[2])) - 1
    g2(x) = C2 * sqrt(1 + x[2]^2) * (8 / x[1] - 1 / (x[1] * x[2])) - 1
    dg1(x) = [-C2 * (8 * x[2] + 1) * sqrt(1 + x[2]^2) / (x[1]^2 * x[2]), C2 * (8 * x[2]^3 - 1) / (x[1] * x[2]^2 * sqrt(1 + x[2]^2))]
    dg2(x) = [-C2 * (8 * x[2] - 1) * sqrt(1 + x[2]^2) / (x[1]^2 * x[2]), C2 * (8 * x[2]^3 + 1) / (x[1] * x[2]^2 * sqrt(1 + x[2]^2))]
    constraint = MMA.Constraints([g1, g2], [dg1, dg2])

    prob = MMA.MMAProblem(x0, objective, constraint)
    for _ = 1:100
        MMA.iterate(prob)
        MMA.relative_change(prob.state) < 1e-8 && break
    end
    @test prob.state.x ≈ [1.41, 0.38] atol = 1e-2
    @test prob.state.cur_obj ≈ 1.51 atol = 1e-2
end
