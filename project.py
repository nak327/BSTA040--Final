import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import scipy.stats as stats
import streamlit as st

#read csv file 
ilidf = pd.read_csv("ilidata.csv")

#Title
st.header ("BSTA 040 Final - Natalie Kam")

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
st.write (""" The line chart displays the trend of Influenza-like Illness (ILI) percentages over time for the selected states. The x-axis represents the weeks starting from 0 up to the total number of weeks in the dataset, while the y-axis shows the percentage of ILI reported in those weeks. Each point on the line corresponds to a particular week’s ILI data for the chosen states. As users select different states, this chart dynamically updates to show the time series for the selected regions. The chart provides an insightful view into how the ILI percentage fluctuates over time, which could be influenced by seasonal trends, public health interventions, or other factors. The trend observed in the chart can offer useful information on the temporal behavior of ILI for public health monitoring. """)

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
    st.write(""" This histogram visualizes the distribution of Influenza-like Illness (ILI) percentages, which have been calculated for the selected states over time. The x-axis represents the percentage of ILI, while the y-axis shows the density of the data at each ILI percentage range. The histogram is overlayed with a fitted exponential distribution, represented by the red curve. The exponential distribution is chosen as a theoretical model to fit the data, assuming the ILI percentages follow a decay pattern that is often seen in epidemiological data. The rate parameter (λ) of the exponential fit is estimated as the inverse of the mean of the ILI data. This overlay helps to visualize how well the ILI data matches the exponential distribution, which is often used in survival analysis or modeling the time between events. The histogram allows for easy comparison between the actual ILI data and the theoretical exponential model, providing insights into whether the data fits this type of distribution. """)

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
        This plot demonstrates the Law of Large Numbers (LLN) by showing how the sample mean of ILI percentages converges to the true mean as the sample size increases. In the plot, the x-axis represents the sample mean of the ILI percentages for random samples of increasing size (10, 50, 100, 500), while the y-axis represents the density of those sample means. For each sample size, we simulate 1,000 random samples and compute the mean of ILI percentages for each sample. As the sample size increases, the distribution of the sample means narrows and concentrates around the true mean (shown by the red dashed line). This graph vividly illustrates the LLN concept, which states that as the sample size increases, the sample mean becomes a better estimate of the population mean. For smaller sample sizes (10 flips), the sample mean varies more widely, while for larger sample sizes (500 flips), the mean is much closer to the true population mean, confirming the LLN’s expected behavior. """)