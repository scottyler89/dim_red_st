import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from experiment import (
    generate_data,
    dim_reduction,
    plot_dim_reductions,
    plot_obs_data_heatmap,
    plot_intrinsic_dimensionality,
    plot_distance_correlations,
    pca_wrapper,
    nmf_wrapper,
    tsne_wrapper,
    umap_wrapper,
    som_wrapper,
    true_gen_func,
    redundant_gen_noise_func
)
import skdim

# Sidebar for setting parameters
st.sidebar.header("Set Experiment Parameters")
n_obs = st.sidebar.select_slider("Number of observations:", options=[
                                 100,500,1000,3000])
true_dims = 2#st.sidebar.slider("Number of true dimensions:", 1, 10, 2)
n_redundant_per_true = st.sidebar.slider(
    "Number of redundant dimensions per true dimension:", 1, 200, 100)


################################
# let the user define the sd ratios
# A session state to keep track of the added values
if 'sd_ratios' not in st.session_state:
    st.session_state.sd_ratios = [0.01,0.05,.25,1.0]

# Allow the user to input a floating-point value within the range
new_sd_ratio = st.sidebar.number_input(
    "Add an SD Ratio (interesting range is 0.01-1.0):",
    min_value=0.01, max_value=5.0, step=0.01, value=0.01)

# Button to add the entered value to the list
if st.sidebar.button("Add SD Ratio"):
    st.session_state.sd_ratios.append(new_sd_ratio)
    st.session_state.sd_ratios = list(
        map(float,sorted(list(set(
        st.session_state.sd_ratios)))))  # Sort the values

# Button to clear the SD Ratios
if st.sidebar.button("Clear SD Ratios"):
    st.session_state.sd_ratios = []


# Display the current SD Ratios
st.sidebar.write("Selected SD Ratios:")
st.sidebar.write(st.session_state.sd_ratios)


#sd_ratios = st.sidebar.multiselect("SD Ratios:", [
#                                    0.01, 0.05, 0.1, 0.25, 0.5, 1.], [0.01, 0.05, 0.25, 0.5, 1.])
#######################


sep = st.sidebar.select_slider(
    "Separation Vector:", options=[0, 2, 4, 6, 8])

# Button to run the experiment
run_button = st.sidebar.button("Run Experiment")

# Create a placeholder for the running message
running_message = st.empty()


# Display the results when the button is pressed
if run_button:
    # Display the running message
    running_message.write(
        "The experiment is running.\n\nThis may take ~15 minutes depending on the parameters...")

    # Setting the random seed
    np.random.seed(123456)

    # Defining dictionaries to store results
    sep_dict = {}
    intrinsic_dim_estimate_dict = {}
    true_dim_dict = {}
    obs_data_dict = {}
    results_dict = {}
    sd_lookup = {}

    # Iterating through separation values

    sep_name = "Clust Sep:" + str(sep)
    for sd_ratio in st.session_state.sd_ratios:
        sd_ratio = float(sd_ratio)
        sd_name = "SD ratio:" + str(sd_ratio)
        sd_lookup[sd_name] = sd_ratio
        final_dims = true_dims

        # Generate data
        true_dim_data, obs_data = generate_data(
            n_obs, true_dims, n_redundant_per_true, true_gen_func, redundant_gen_noise_func, sd_ratio, separation=sep)

        # Estimate intrinsic dimensionality
        danco = skdim.id.DANCo().fit(obs_data)
        intrinsic_dim_estimate_dict[sd_name] = danco.dimension_

        # Log the data
        true_dim_dict[sd_name] = true_dim_data
        obs_data_dict[sd_name] = obs_data

        # Perform dimension reduction
        dim_red_funcs = [pca_wrapper, nmf_wrapper,
                            tsne_wrapper, umap_wrapper, som_wrapper]
        dim_red_names = ["PCA", "NMF", "tSNE", "UMAP", "SOM"]
        results_dict[sd_name] = dim_reduction(
            obs_data, dim_red_funcs, dim_red_names, final_dims)

    # Clear the running message placeholder
    running_message.empty()
    # Call the plotting functions
    st.write(f"### Results for Separation = {sep}")
    st.write("#### True Dimensions with Noise vs Dimension Reduction")
    fig1 = plot_dim_reductions(true_dim_dict, results_dict, "")
    st.pyplot(fig1)

    st.write("#### Heatmap and Scatter Plots")
    fig2 = plot_obs_data_heatmap(
        true_dim_dict, obs_data_dict, intrinsic_dim_estimate_dict, "")
    st.pyplot(fig2)

    st.write("#### Intrinsic Dimensionality vs Noise Level")
    fig3 = plot_intrinsic_dimensionality(
        sd_lookup, intrinsic_dim_estimate_dict, "")
    st.pyplot(fig3)

    st.write("#### Distance Correlations")
    fig4 = plot_distance_correlations(true_dim_dict, results_dict, "")
    st.pyplot(fig4)

    sep_dict[sep_name] = results_dict
