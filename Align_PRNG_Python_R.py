import numpy as np


def int32(num):
    while num > 2**31 or num < -2**31:
        if num > 2 ** 31:
            num = -(2 ** 32 - num)
        elif num < -2 ** 31:
            num = 2 ** 32 + num
    return num


def int32_rev(num):
    if num < 0:
        return 2**32+num
    return num


def state32(state):
    pos, key = state
    return pos, [int32(x) for x in key]


n = 624
m = 397
matrix_a = 0x9908b0df
upper_mask = 0x80000000
lower_mask = 0x7fffffff


def mt19937(key):
    for i in range(n - m):
        y = (key[i] & upper_mask) | (key[i + 1] & lower_mask)
        key[i] = key[i + m] ^ (y >> 1) ^ (-(y & 1) & matrix_a)
    for i in range(n - m, n - 1):
        y = (key[i] & upper_mask) | (key[i + 1] & lower_mask)
        key[i] = key[i + (m - n)] ^ (y >> 1) ^ (-(y & 1) & matrix_a)
    y = (key[n - 1] & upper_mask) | (key[0] & lower_mask)
    key[n - 1] = key[m - 1] ^ (y >> 1) ^ (-(y & 1) & matrix_a)
    return key


def update_state(state):
    pos = state[0]
    key = state[1]
    if pos >= 623:
        return 0, mt19937(key)
    else:
        return pos + 1, key


def mt19937_next(state):
    pos, key = update_state(state)
    y = key[pos]
    y = int32_rev(y)
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= (y >> 18)
    return y, (pos, key)


def r_state_from_seed(seed):
    for j in range(51):
        seed = int32(69069*seed+1)
    key = []
    for i in range(624):
        seed = int32(69069*seed+1)
        key.append(seed)
    return 624, key


def r_random_uniform(state, size=1, a=0, b=1):
    output = []
    for i in range(size):
        state = state32(state)
        integer_output, state = mt19937_next(state)
        output.append(a+(b-a)*integer_output/(2**32-1))
    return output, state


def r_random_exponential(state, size=1):
    output = []

    def r_random_exponential_variate(state):
        q = [
            0.6931471805599453,
            0.9333736875190459,
            0.9888777961838675,
            0.9984959252914960,
            0.9998292811061389,
            0.9999833164100727,
            0.9999985691438767,
            0.9999998906925558,
            0.9999999924734159,
            0.9999999995283275,
            0.9999999999728814,
            0.9999999999985598,
            0.9999999999999289,
            0.9999999999999968,
            0.9999999999999999,
            1.0000000000000000
            ]
        a = 0
        u, state = r_random_uniform(state)
        u1 = u[0]
        while u1 <= 0 or u1 >= 1:
            u, state = r_random_uniform(state)
            u1 = u[0]
        while True:
            u1 += u1
            if u1 > 1:
                break
            a += q[0]
        u1 -= 1
        if u1 <= q[0]:
            return a+u1, state
        i = 0
        ustar, state = r_random_uniform(state)
        ustar1 = ustar[0]
        umin = ustar1
        while u1 > q[i]:
            ustar, state = r_random_uniform(state)
            ustar1 = ustar[0]
            if umin > ustar1:
                umin = ustar1
            i += 1
        return a+umin*q[0], state
    for k in range(size):
        e1, state = r_random_exponential_variate(state)
        output.append(e1)
    return output, state


def r_random_normal(state, mu=0, sigma=1, size=1):
    import numpy as np
    output = []

    def r_qnorm(p, mu, sigma, lower_tail, log_p):

        def R_D_Cval(p, lower_tail):
            if lower_tail:
                return 0.5-p+0.5
            else:
                return p

        def R_D_qIv(p, log_p):
            if log_p:
                return np.exp(p)
            else:
                return p

        def R_D_Lval(p, lower_tail):
            if lower_tail:
                return p
            else:
                return 0.5-p+0.5

        def R_DT_qIv(p, lower_tail, log_p):
            return R_D_Lval(R_D_qIv(p, log_p), lower_tail)

        p_ = R_DT_qIv(p, lower_tail, log_p)
        q = p_-0.5
        if abs(q) <= 0.425:
            r = 0.180625 - q**2
            val = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r
                           + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                        + 133.14166789178437745) * r + 3.387132872796366608) /\
                  (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r
                       + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                    + 42.313330701600911252) * r + 1.)
        else:
            if log_p and ((lower_tail and q <= 0) or (not lower_tail and q > 0)):
                r = p
            else:
                if q > 0:
                    r = np.log(R_D_Cval(R_D_qIv(p, log_p), lower_tail))
                else:
                    r = np.log(p_)
            r = np.sqrt(-r)
            if r <= 5:
                r += (-1.6)
                val = (((((((r * 7.7454501427834140764e-4 + .0227238449892691845833) * r + .24178072517745061177) * r
                           + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                        + 4.6303378461565452959) * r + 1.42343711074968357734) / \
                      (((((((r * 1.05075007164441684324e-9 + 5.475938084995344946e-4) * r + .0151986665636164571966) * r
                           + .14810397642748007459) * r + .68976733498510000455) * r + 1.6763848301838038494) * r
                        + 2.05319162663775882187) * r + 1.)
            elif r >= 816:
                val = r * 1.41421356237309504880
            else:
                r += (-5)
                val = (((((((r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5) * r + .0012426609473880784386)
                           * r + .026532189526576123093) * r + .29656057182850489123) * r + 1.7848265399172913358)
                        * r + 5.4637849111641143699) * r + 6.6579046435011037772) / \
                      (((((((r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7)
                            * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r
                          + .0148753612908506148525) * r + .13692988092273580531) * r + .59983220655588793769) * r + 1.)
            if q < 0:
                val = -val
        return mu + sigma * val

    def r_random_normal_variate(state):
        big = 134217728
        u1, state = r_random_uniform(state)
        u2, state = r_random_uniform(state)
        u = int(big*u1[0]) + u2[0]
        return mu+sigma*r_qnorm(u/big, mu, sigma, True, False), state
    for k in range(size):
        n1, state = r_random_normal_variate(state)
        output.append(n1)
    return output, state


def r_rbits(bits, state):
    v = 0
    n = 0
    while n <= bits:
        u1, state = r_random_uniform(state)
        v1 = int(np.floor(u1[0]*65536))
        v = 65536*v+v1
        n+=16
    return v&((1 << bits)-1), state


def r_R_unif_index(dn, state):
    if dn <= 0:
        return 0, state
    bits = int(np.ceil(np.log2(dn)))
    dv, state = r_rbits(bits, state)
    while dn <= dv:
        dv, state = r_rbits(bits, state)
    return dv, state


def r_random_sample(n, size, replace, state, p=None):
    if replace:
        if p is not None:
            def r_random_sample_replace_prob(n, p, size, state):
                perm = np.empty(n)
                ans = np.empty(size, dtype=int)
                nm1 = n-1
                for i in range(n):
                    perm[i]=i
                for i in range(1,n):
                    p[i]+=p[i-1]
                for i in range(size):
                    rU, state = r_random_uniform(state)
                    for j in range(nm1):
                        if rU[0]<=p[j]:
                            break
                    if rU[0]>p[j]:
                        j=nm1
                    ans[i]=(perm[j]+1)%n
                return ans, state
            return r_random_sample_replace_prob(n, p, size, state)
        else:
            def r_random_sample_replace_noprob(n, size, state):
                y = np.empty(size, dtype=int)
                for i in range(size):
                    j, state = r_R_unif_index(n, state)
                    y[i]=j
                return y, state
            return r_random_sample_replace_noprob(n, size, state)
    else:
        if p is not None:
            def r_random_sample_noreplace_prob(n, size, p, state):
                perm = np.empty(n)
                ans = np.empty(size, dtype=int)
                for i in range(n):
                    perm[i]=i
                totalmass=1
                n1=n-1
                for i in range(size):
                    u, state = r_random_uniform(state)
                    rT= totalmass * u[0]
                    mass=0
                    for j in range(n1):
                        mass+=p[j]
                        if rT<=mass:
                            break
                    if rT>mass:
                        j=n1
                    ans[i]=(perm[j]+1)%n
                    totalmass-=p[j]
                    for k in range(j,n1):
                        p[k]=p[k+1]
                        perm[k]=perm[k+1]
                    n1-=1
                return ans, state
            return r_random_sample_noreplace_prob(n, size, p, state)
        else:
            def r_random_sample_noreplace_noprob(n, size, state):
                x = np.empty(n)
                y = np.empty(size, dtype=int)
                for i in range(n):
                    x[i] = i
                N=n
                for k in range(size):
                    j, state = r_R_unif_index(N, state)
                    y[k]=x[j]
                    N-=1
                    x[j] = x[N]
                return y, state
            return r_random_sample_noreplace_noprob(n, size, state)
