    # 2) Fit 3 esponenziali usando popt1 per partire vicino
    tau1_est = popt1[2] * 2  # da sigma→τ1
    p0_2 = [
        popt1[0],    # A1
        tau1_est,    # tau1
        popt1[3],    # A2
        popt1[4],    # tau2
        popt1[5],    # A3
        popt1[6],    # tau3
        200,         # L
        popt1[7]     # x0
    ]
    bounds_2 = (
        [0,   1,   0,   10,  0,   10,  10,  popt1[7]],
        [np.inf, 200, np.inf, 500, np.inf, 2000, 2000, popt1[7]+1]
    )
    popt2, pcov2 = curve_fit(
        model_3exp, x, avg_cosmic,
        p0=p0_2, bounds=bounds_2
    )


    -------------------------------------------


    A_int_2exp = popt1[3]
    tau_int_2exp = popt1[4]
    tau_slow_2exp = popt1[6]
    L_init = 200
    x0_2   = popt1[7]

    # new initial guesses
    p0_2 = [
      0.3*A_int_2exp,  0.3*tau_int_2exp,
      0.5*A_int_2exp,      tau_int_2exp,
      0.2*A_int_2exp,      tau_slow_2exp,
      L_init,     x0_2
    ]

    bounds_2 = (
        [0,   1,   0,   10,  0,   10,  10,  popt1[7]],
        [np.inf, 200, np.inf, 500, np.inf, 2000, 2000, popt1[7]+1]
    )

    popt2, pcov2 = curve_fit(
      model_3exp, x, avg_cosmic,
      p0=p0_2, bounds=bounds_2
    )
    print("3-exp amplitudes:", popt2[0], popt2[2], popt2[4])
