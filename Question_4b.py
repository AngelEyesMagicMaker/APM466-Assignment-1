def compute_forward_rates(spot_rates, maturities):
    """
    Compute forward rates from spot rates.
    """
    forward_rates = []
    for i in range(1, len(spot_rates)):
        t1 = maturities[i - 1]
        t2 = maturities[i]
        r1 = spot_rates[i - 1]
        r2 = spot_rates[i]

        # Forward rate calculation
        fwd_rate = ((1 + r2) ** t2 / (1 + r1) ** t1) ** (1 / (t2 - t1)) - 1
        forward_rates.append(fwd_rate)

    return forward_rates

# Compute Forward Rates
forward_rates = compute_forward_rates(spot_rates, maturities)

# Create DataFrame for display
forward_rate_df = pd.DataFrame({
    "Start Maturity (Years)": maturities[:-1],
    "End Maturity (Years)": maturities[1:],
    "Forward Rate": forward_rates
})

# Print the results
print(forward_rate_df)
