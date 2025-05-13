using CSV
using DataFrames
using Impute
using StatsBase
using Random
using PrettyTables
using MLJ
using MLJBase

# Load the dataset
data = CSV.read("diabetic_retinopathy.csv", DataFrame)

# Clean column names by removing leading/trailing whitespace
rename!(data, Symbol.(strip.(string.(names(data)))))

# Display basic information using PrettyTables
println("Dataset Info:")
pretty_table(describe(data); backend=Val(:text), tf=tf_unicode, show_row_number=true)
println("First few rows:")
pretty_table(first(data, 5); backend=Val(:text), tf=tf_unicode, show_row_number=true)

# Check class distribution using PrettyTables
println("Class Distribution:")
class_dist = combine(groupby(data, :Clinical_Group), nrow => :count)
pretty_table(class_dist; backend=Val(:text), tf=tf_unicode, show_row_number=true)

# Replace "NIL" and "Nil" with missing
for col in names(data)
    data[!, col] = replace(data[!, col], "NIL" => missing, "Nil" => missing, "NaN" => missing)
end

# Separate numerical and categorical columns
numerical_cols = [:Hornerin, :SFN, :Age, :Diabetic_Duration, :eGFR, :HB, :EAG, :FBS, :RBS, :HbA1C, 
                  :Systolic_BP, :Diastolic_BP, :BUN, :Total_Protein, :Serum_Albumin, :Serum_Globulin, 
                  :AG_Ratio, :Serum_Creatinine, :Sodium, :Potassium, :Chloride, :Bicarbonate, :SGOT, 
                  :SGPT, :Alkaline_Phosphatase, :T_Bil, :D_Bil, :HDL, :LDL, :CHOL, :Chol_HDL_ratio, :TG]
categorical_cols = [:Gender, :Albuminuria]

# Impute numerical columns with srs
for col in numerical_cols
    if eltype(data[!, col]) <: Union{Missing, Number}
        rng = MersenneTwister(42)
        data[!, col] = Impute.srs(data[!, col], rng=rng)
    end
end

# Impute categorical columns with mode
for col in categorical_cols
    if eltype(data[!, col]) <: Union{Missing, String}
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

# Display the encoded DataFrame using PrettyTables
println("Encoded Dataset (first 5 rows):")
pretty_table(first(data_encoded, 5); 
             backend=Val(:text), 
             tf=tf_unicode, 
             show_row_number=true, 
             alignment=:c, 
             hlines=:all, 
             vlines=:all)