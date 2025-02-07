import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Load the Bond Data
file_path = "/mnt/data/Bond.csv"
raw_bond_data = pd.read_csv(file_path)

# Transform Dataset (Transpose and Extract Relevant Data)
transposed_bond_data = raw_bond_data.set_index(raw_bond_data.columns[0]).T
transposed_bond_data.reset_index(inplace=True)
transposed_bond_data.columns.name = None

# Create Cleaned Bond DataFrame
cleaned_bond_data = pd.DataFrame()
cleaned_bond_data["ISIN"] = transposed_bond_data["ISIN"]
cleaned_bond_data["Market Price"] = pd.to_numeric(transposed_bond_data["Issue Price       (Market Price)"], errors='coerce')
cleaned_bond_data["Issue Date"] = pd.to_datetime(transposed_bond_data["Issue Date"], errors='coerce')
cleaned_bond_data["Maturity (Years)"] = 2025 - cleaned_bond_data["Issue Date"].dt.year
cleaned_bond_data["Face Value"] = 100  # Assume face value is 100
cleaned_bond_data["Coupon Rate"] = 0.05  # Assume 5% coupon rate if missing
cleaned_bond_data = cleaned_bond_data.dropna()  # Drop missing values

# Function to Compute Yield to Maturity (YTM)
def safe_ytm(price, face_value, coupon_rate, years, freq=2):
    """
    Computes Yield to Maturity (YTM) safely, avoiding errors from invalid inputs.
    """
    if price <= 0 or years <= 0:
        return np.nan

    try:
        ytm_solution = fsolve(lambda ytm: sum([(coupon_rate / freq * face_value) / (1 + ytm / freq) ** (freq * t)
                                               for t in range(1, int(years * freq) + 1)]) +
                                          face_value / (1 + ytm / freq) ** (freq * years) - price,
                              0.05)  # Initial guess: 5%
        return float(ytm_solution[0])
    except:
        return np.nan

# Compute YTM for Each Bond
cleaned_bond_data["YTM"] = cleaned_bond_data.apply(
    lambda row: safe_ytm(row["Market Price"], row["Face Value"], row["Coupon Rate"], row["Maturity (Years)"]), axis=1
)

# Function to Bootstrap the Spot Rate Curve
def bootstrap_spot_rates(bond_data):
    """
    Uses bootstrapping to derive the spot rate curve from bond data.
    """
    bond_data = bond_data.sort_values(by="Maturity (Years)")
    spot_rates = {}

    for i, row in bond_data.iterrows():
        maturity = row["Maturity (Years)"]
        price = row["Market Price"]
        coupon = row["Coupon Rate"] * row["Face Value"]
        face_value = row["Face Value"]

        if maturity == 1:  # First year, zero-coupon approximation
            spot_rates[maturity] = (face_value + coupon) / price - 1
        else:
            # Solve for spot rate iteratively using previous spot rates
            total_coupon_value = sum([coupon / (1 + spot_rates[t]) ** t for t in spot_rates])
            spot_rates[maturity] = ((price - total_coupon_value) / face_value) ** (1 / maturity) - 1

    return spot_rates

# Compute Spot Rate Curve
spot_rates = bootstrap_spot_rates(cleaned_bond_data)

# Function to Compute Forward Rates from Spot Rates
def compute_forward_rates(spot_rates):
    """
    Computes forward rates using the formula:
    F(t, t+n) = [(1 + S(t+n))^(t+n) / (1 + S(t))^t]^(1/n) - 1
    """
    forward_rates = {}

    for t in range(2, 6):  # Compute forward rates for 2 to 5 years
        if t in spot_rates and 1 in spot_rates:  # Ensure necessary spot rates exist
            forward_rates[f"1yr-{t}yr"] = ((1 + spot_rates[t]) ** t / (1 + spot_rates[1]) ** 1) ** (1 / (t - 1)) - 1

    return forward_rates

# Compute Forward Rate Curve
forward_rates = compute_forward_rates(spot_rates)

# Convert to DataFrames for Plotting
ytm_df = pd.DataFrame({"Maturity (Years)": cleaned_bond_data["Maturity (Years)"], "YTM": cleaned_bond_data["YTM"]})
spot_df = pd.DataFrame({"Maturity (Years)": list(spot_rates.keys()), "Spot Rate": list(spot_rates.values())})
forward_df = pd.DataFrame({"Maturity (Years)": list(forward_rates.keys()), "Forward Rate": list(forward_rates.values())})

# Plot Yield to Maturity (YTM) Curve
plt.figure(figsize=(8, 5))
plt.plot(ytm_df["Maturity (Years)"], ytm_df["YTM"], marker="o", linestyle="-", label="YTM Curve")
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield to Maturity")
plt.title("Yield to Maturity (YTM) Curve")
plt.legend()
plt.grid(True)
plt.show()

# Plot Spot Rate Curve
plt.figure(figsize=(8, 5))
plt.plot(spot_df["Maturity (Years)"], spot_df["Spot Rate"], marker="s", color="red", label="Spot Curve")
plt.xlabel("Maturity (Years)")
plt.ylabel("Spot Rate")
plt.title("Spot Rate Curve")
plt.legend()
plt.grid(True)
plt.show()

# Plot Forward Rate Curve
plt.figure(figsize=(8, 5))
plt.plot(forward_df["Maturity (Years)"], forward_df["Forward Rate"], marker="^", color="green", label="Forward Curve")
plt.xlabel("Maturity (Years)")
plt.ylabel("Forward Rate")
plt.title("Forward Rate Curve")
plt.legend()
plt.grid(True)
plt.show()
