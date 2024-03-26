# %%
# Here, we generate some data using rebound.
import rebound
import numpy as np


# %%
def make_sim(rstate):
    sim = rebound.Simulation()
    sim.add(m=1.0)
    nplanet = rstate.randint(2, 6)

    m = rstate.uniform(1e-5, 5e-5, nplanet)
    a_final = rstate.uniform(1.5, 3.0)
    a = np.linspace(1.0, a_final, nplanet)
    e = rstate.uniform(0.0, 0.1, nplanet)
    theta = rstate.uniform(0.0, 2 * np.pi, nplanet)

    # with probability 0.3, set all theta to 0
    if rstate.uniform() < 0.3:
        theta = np.zeros(nplanet)

    # random permutation of the planet order
    perm = rstate.permutation(nplanet)
    m, a, e, theta = m[perm], a[perm], e[perm], theta[perm]

    for i in range(nplanet):
        sim.add(m=m[i], a=a[i], e=e[i], theta=theta[i])

    return sim, {"m": list(m), "a": list(a), "e": list(e)}


# %%
def integrate(sim, t):
    outputs = []
    for ti in t:
        # Counter-intuitively, sim.integrate
        # integrates from 0 to its input, not
        # from its current time to its input.
        sim.integrate(ti)
        cur_out = []
        for i, particle in enumerate(sim.particles):
            if i == 0:
                continue

            cur_out.append(
                dict(
                    a=particle.a,
                    e=particle.e,
                    # Omega=particle.Omega,  # longitude of ascending node
                    omega=particle.omega,  # argument of periapsis
                    f=particle.f,  # true anomaly
                    # (Just for plotting)
                    x=particle.x,
                    y=particle.y,
                )
            )
        outputs.append(cur_out)
    return outputs


# %%
def construct_example(seed=None):
    rstate = np.random.RandomState(seed)
    sim, meta = make_sim(rstate)
    revs = rstate.randint(15, 25)
    steps = rstate.randint(30, 60)
    t = np.linspace(0, revs, steps)
    data = integrate(sim, t)
    return {
        "data": data,
        "description": {"seed": seed, **meta, "stepsize": t[1] - t[0]},
    }


# %%
