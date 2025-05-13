using CSV
using DataFrames
using Impute
using StatsBase
using Random
using PrettyTables
using MLJ
using MLJBase

# Load the dataset
data = CSV.read("diabetic_retinopathy.csv", DataFrame; missingstring=["NIL", "Nil", "NaN"])

# Clean column names by removing leading/trailing whitespace
rename!(data, Symbol.(strip.(string.(names(data)))))

# Display basic information using PrettyTables with full column display
println("Dataset Info:")
pretty_table(
    describe(data),
    backend=Val(:text),
    tf=tf_unicode,
    show_row_number=true,
    alignment=:c,
    crop=:none                     # Prevent truncation
)

# Display first few rows
println("\nFirst 5 rows of dataset:")
pretty_table(
    first(data, 5),
    backend=Val(:text),
    tf=tf_unicode,
    show_row_number=true,
    alignment=:c,
    crop=:none
)

# Check class distribution
println("\nClass Distribution:")
class_dist = combine(groupby(data, :Clinical_Group), nrow => :count)
pretty_table(
    class_dist,
    backend=Val(:text),
    tf=tf_unicode,
    show_row_number=true,
    alignment=:c,
    crop=:none
)

# Convert columns to appropriate types, handling missing values
for col in names(data)
    if eltype(data[!, col]) <: Union{Missing, String}
        # For numerical columns stored as strings, convert to Float64
        if col in [:HB, :EAG]
            data[!, col] = tryparse.(Float64, string.(data[!, col]))
        end
    end
end

# Separate numerical and categorical columns
numerical_cols = [:Hornerin, :SFN, :Age, :Diabetic_Duration, :eGFR, :HB, :EAG, :FBS, :RBS, :HbA1C, 
                  :Systolic_BP, :Diastolic_BP, :BUN, :Total_Protein, :Serum_Albumin, :Serum_Globulin, 
                  :AG_Ratio, :Serum_Creatinine, :Sodium, :Potassium, :Chloride, :Bicarbonate, :SGOT, 
                  :SGPT, :Alkaline_Phosphatase, :T_Bil, :D_Bil, :HDL, :LDL, :CHOL, :Chol_HDL_ratio, :TG]
categorical_cols = [:Gender, :Albuminuria]

# Impute numerical columns with simple random sampling (srs)
for col in numerical_cols
    if any(ismissing, data[!, col])
        try
            rng = MersenneTwister(42)
            data[!, col] = Impute.srs(data[!, col]; rng=rng)
        catch e
            println("Warning: Could not impute column $col: $e")
            # Fallback to mean imputation for numerical columns
            mean_val = mean(skipmissing(data[!, col]))
            data[!, col] = coalesce.(data[!, col], mean_val)
        end
    end
end

# Impute categorical columns with mode
for col in categorical_cols
    if any(ismissing, data[!, col])
        mode_val = mode(skipmissing(data[!, col]))
        data[!, col] = coalesce.(data[!, col], mode_val)
    end
end

# Encode categorical variables
data[!, :Clinical_Group] = categorical(data[!, :Clinical_Group])
hot_encoder = OneHotEncoder(; features=[:Gender, :Albuminuria], drop_last=false)
mach = machine(hot_encoder, data)
fit!(mach)
data_encoded = MLJBase.transform(mach, data)
data_encoded = DataFrames.select(data_encoded, Not([:Gender, :Albuminuria]))

# Display the encoded DataFrame
println("\nEncoded Dataset (first 5 rows):")
pretty_table(
    first(data_encoded, 5),
    backend=Val(:text),
    tf=tf_unicode,
    show_row_number=true,
    alignment=:c,
    crop=:none
)