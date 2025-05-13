# Load required packages
using DataFrames
using CSV
using MLJ
using Plots
using Measures  # Required for mm unit in plot margins

# Step 1: Create a DataFrame with breast cancer data for 2025
data = DataFrame(
    Category = [
        "Invasive (Women)",
        "DCIS (Women)",
        "Deaths (Women)",
        "Invasive (Men)",
        "Deaths (Men)",
        "Cases (Black Women)"
    ],
    Number = [316950, 59080, 42170, 2800, 510, 40530]
)

# Step 2: Save the DataFrame to a CSV file
CSV.write("breast_cancer_2025.csv", data)

# Step 3: Load the CSV file back into a DataFrame
loaded_data = CSV.read("breast_cancer_2025.csv", DataFrame)

# Step 4: Use MLJ to compute valid summary statistics on the Number column
schema(loaded_data)  # Display schema of the DataFrame
describe_stats = describe(loaded_data, cols=[:Number], :mean, :median, :min, :max, :std)
println("Summary Statistics for Number column:")
println(describe_stats)

# Step 5: Create a bar plot using Plots.jl
bar_plot = bar(
    loaded_data.Category,
    loaded_data.Number,
    title = "Breast Cancer Statistics in the U.S. (2025 Estimates)",
    ylabel = "Number of Cases/Deaths",
    xlabel = "Category",
    legend = false,
    bar_width = 0.15,
    xticks = (1:length(loaded_data.Category), loaded_data.Category),
    rotation = 45,  # Rotate x-axis labels for readability
    size = (800, 600),
    margin = 10mm  # Requires Measures.jl
)

# Display the plot
display(bar_plot)

# Save the plot to a file
savefig(bar_plot, "breast_cancer_2025_stats.png")

# Save summary statistics to a CSV file
CSV.write("breast_cancer_2025_stats_summary.csv", describe_stats)