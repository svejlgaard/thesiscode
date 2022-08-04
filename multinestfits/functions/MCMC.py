import emcee


def main(p0,nwalkers,niter,ndim,lnprob,data,func,prior,pool):
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[*data,func,prior], pool=pool)

        print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 10, progress=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

        return sampler, pos, prob, state
