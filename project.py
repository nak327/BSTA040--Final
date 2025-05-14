import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import scipy.stats as stats
import streamlit as st

#read csv file 
ilidf = pd.read_csv("ilidata.csv")

#adding columncounts week from 0 up to total number of weeks in df 
ilidf['weeks'] = range(len(ilidf))

#add selectbox for state selection
states = ilidf['state'].unique()  # Get the list of unique states
selected_states = st.multiselect("Select locations to display charts for:", states)
statedf = ilidf[ilidf['state'].isin(selected_states)]

#plot graph for weeks vs percent ili
st.subheader(f"Line chart for: {', '.join(selected_states)} vs Percent ILI")
st.line_chart(statedf[['weeks', 'ili']].set_index('weeks'))

#Description 
st.write (""" This chart shows the percentage of Influenza-like Illness (ILI) over time (weeks) for the selected state. """)

#Histogram title
st.subheader("Histogram of ILI Percent with Exponential Density")

# Remove NaN values in ILI column
ili_data = statedf['ili'].dropna()

if ili_data.empty:
    st.error("No valid ILI data available for the selected states.")
else:
    # Plot histogram of ILI percentages
    plt.figure(figsize=(10, 6))
    plt.hist(ili_data, bins=30, density=True, alpha=0.6, color='g', label='ILI Histogram')

    # Fit an exponential distribution to the ILI data
    rate = 1 / np.mean(ili_data)  # Calculate the rate parameter (1/mean)
    
    # Ensure ili_data has valid min and max for linspace
    x = np.linspace(min(ili_data), max(ili_data), 1000)  # Generate x-values
    pdf = stats.expon.pdf(x, scale=1 / rate)  # Exponential PDF

    # Overlay the exponential distribution
    plt.plot(x, pdf, 'r-', lw=2, label=f'Exponential fit (λ={rate:.2f})')

    #Add labels and title
    plt.title('Histogram of ILI Percent with Exponential Density Overlay')
    plt.xlabel('ILI Percent')
    plt.ylabel('Density')
    plt.legend()

    # Show the plot in Streamlit
    st.pyplot(plt)

    #Add description for the histogram and exponential fit
    st.write("""
        The histogram represents the distribution of Influenza-like Illness (ILI) percentages over time for the selected state.
        The red curve represents the fitted exponential density function. The parameter 'λ' (rate) is estimated as the inverse of the mean of the ILI data.
    """)

    #Title for LLN 
    st.subheader("Convergence of sample mean to the trume mean (LLN)")
    # Apply the Law of Large Numbers (LLN)
    # True mean of the ILI data
    true_mean = np.mean(ili_data)

    # Number of simulations
    num_simulations = 1000

    # Different sample sizes
    sample_sizes = [10, 50, 100, 500]

    # Create a plot for LLN
    plt.figure(figsize=(10, 6))

    # Loop through each sample size
    for size in sample_sizes:
        sample_means = []
        for _ in range(num_simulations):
            # Take a random sample of the given size
            sample = np.random.choice(ili_data, size=size, replace=False)
            sample_means.append(np.mean(sample))

        # Plot the sample means for each size
        plt.hist(sample_means, bins=30, density=True, alpha=0.6, label=f'Sample Size {size}')

    # Add the true mean line
    plt.axvline(true_mean, color='r', linestyle='--', label=f'True Mean = {true_mean:.2f}')

    # Add labels and title
    plt.title("Convergence of Sample Mean to the True Mean (LLN)")
    plt.xlabel("Sample Mean")
    plt.ylabel("Density")
    plt.legend()

    # Show the plot in Streamlit
    st.pyplot(plt)

    # Add description for LLN plot
    st.write("""
        The plot shows how the sample mean of ILI percentages (shown in histogram) converges to the true mean (represented by the exponential density function) as the sample size increases.
        This demonstrates the Law of Large Numbers (LLN): as the sample size increases, the sample mean gets closer to the true population mean.
    """)