# Student Enrollment Number: 25472
# Partner Enrollment Number: 50833

# Assignment 1 - 7313

# Importing modules
from sqlalchemy import create_engine, text
import json
import pandas as pd
import time

# Database connection configuration
host = "mysql-1.cda.hhs.se"
username = "7313"
password = "data"
schema = "MortgageApplications"

# Fixed input variables
CHILD_COST_PER_MONTH = 3700
STATE_TAX_THRESHOLD = 643100
STATE_TAX_RATE = 0.2
MAXIMUM_INTEREST_RATE = 0.20
ADULT_COST_OF_DAILY_LIVING_PER_MONTH = 10000
HOUSE_COST_PER_MONTH = 4700
APARTMENT_COST_PER_MONTH = 4200
INTEREST_RATE_FROM_BANK = 0.07
MONTHS_OF_A_YEAR = 12
INTEREST_RATE_SAFETY_MARGIN = 0.03
MARGIN_OF_ERROR = 100
MAX_PERCENTAGE_OF_PROPERTY_VALUATION = 0.85

def load_data():
    """Load and prepare data from MySQL database"""
    print("Loading data from database...")
    
    #Sets start time to current time
    start_time = time.time()
    
    # Create database connection
    connection_string = "mysql+pymysql://{}:{}@{}/{}".format(username, password, host, schema)
    engine = create_engine(connection_string)
    
    with engine.connect() as connection:
        # Load all required tables into dataframes
        customers_df = pd.read_sql_query(con=connection, sql=text("SELECT * FROM Customer"))
        applications_df = pd.read_sql_query(con=connection, sql=text("SELECT * FROM LoanApplication"))
        tax_rates_df = pd.read_sql_query(con=connection, sql=text("SELECT * FROM TaxRate"))
        existing_loans_df = pd.read_sql_query(con=connection, sql=text("SELECT * FROM CustomerLoan"))
    
    #Tells the user how many customers, applications, tax rates, and existing loans were loaded
    print(f"Loaded {len(customers_df)} customers, {len(applications_df)} applications, {len(tax_rates_df)} tax rates, {len(existing_loans_df)} existing loans")
    
    # Get latest tax rates for each municipality
    latest_tax_rates = tax_rates_df.loc[tax_rates_df.groupby('municipality')['tax_year'].idxmax()]

    #Sets the municipality as the index and the tax rate as the value in a dictionary
    tax_lookup = latest_tax_rates.set_index('municipality')['tax_rate'].to_dict()
    
    # Joins the main data by customer_id
    merged_data = pd.merge(applications_df, customers_df, on='customer_id', how='inner')
    
    # Adds tax rates based on the municipality of the customer (using the dictionary)
    merged_data['tax_rate'] = merged_data['municipality'].map(tax_lookup) / 100
    
    # Convert housing type to uppercase to match expected values ('HOUSE' or 'APARTMENT'), a system in place to eliminate case sensitivity
    merged_data['housing_type'] = merged_data['housing_type'].str.upper()
    
    # Ensures numeric types as pandas can miss converting strings to numbers, coerse = NaN if conversion fails
    numeric_cols = ['property_valuation', 'requested_loan', 'gross_yearly_income', 'tax_rate', 'num_children']
    for col in numeric_cols:
        merged_data[col] = pd.to_numeric(merged_data[col], errors='coerce')
    
    # Calculation of cost for existing loans, clip is used to ensure the interest rate is below the maximum interest rate (20%)
    existing_loans_df['adjusted_rate'] = (existing_loans_df['interest_rate'] + INTEREST_RATE_SAFETY_MARGIN).clip(upper=MAXIMUM_INTEREST_RATE)
    existing_loans_df['annual_cost'] = existing_loans_df['amount'] * existing_loans_df['adjusted_rate']
    
    # Group by customer and sum costs, dictionary is used to be efficient in lookup
    existing_loans_cost_lookup = existing_loans_df.groupby('customer_id')['annual_cost'].sum().to_dict()
    
    #Status update for the user/ensuring it is working
    print(f"Merged dataset: {len(merged_data)} records")
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    
    return merged_data, existing_loans_cost_lookup

def amortization_rate(property_valuation, requested_loan, gross_yearly_income):
    """Calculate amortization rate"""
    amortization_rate = 0.0
    
    loan_to_value = requested_loan / property_valuation
    gross_annual_income_factor = gross_yearly_income * 4.5
    
    if loan_to_value > 0.5:
        amortization_rate += 0.01
    if loan_to_value > 0.7:
        amortization_rate += 0.01
    if requested_loan > gross_annual_income_factor:
        amortization_rate += 0.01
    
    return amortization_rate

def calculate_child_cost(num_children):
    """Calculate annual child cost"""
    child_cost_lookup = {0: 0, 1: 1250, 2: 2650, 3: 4480, 4: 6740, 5: 9240, 6: 11740}
    
    if num_children >= 7:
        monthly_net_child_cost = 11740 + (num_children - 6) * 1250
    else:
        monthly_net_child_cost = child_cost_lookup.get(num_children, 0)
    
    monthly_net_child_cost = CHILD_COST_PER_MONTH * num_children - monthly_net_child_cost
    annual_net_child_cost = monthly_net_child_cost * MONTHS_OF_A_YEAR
    
    return annual_net_child_cost

def calculate_taxes(gross_yearly_income, tax_rate):
    """Calculate income tax"""
    income_tax = gross_yearly_income * tax_rate
    
    if gross_yearly_income > STATE_TAX_THRESHOLD:
        income_tax += (gross_yearly_income - STATE_TAX_THRESHOLD) * STATE_TAX_RATE
    
    return income_tax

def calculate_annual_housing_cost(housing_type):
    """Calculate annual housing cost"""
    if housing_type == 'HOUSE':
        return HOUSE_COST_PER_MONTH * MONTHS_OF_A_YEAR
    elif housing_type == 'APARTMENT':
        return APARTMENT_COST_PER_MONTH * MONTHS_OF_A_YEAR
    else:
        return 0 # Default to 0 if the housing type is not 'HOUSE' or 'APARTMENT'

def get_existing_loans_cost(customer_id, existing_loans_lookup):
    """Get pre-calculated existing loans cost"""
    customer_existing_loans_cost = existing_loans_lookup.get(customer_id, 0)

    return customer_existing_loans_cost

def calculate_annual_disposable_income(row, loan_amount, existing_loans_lookup):
    """Calculate annual disposable income"""
    # Calculate amortization cost
    amort_rate = amortization_rate(
        row['property_valuation'], loan_amount, row['gross_yearly_income']
    )
    annual_amortization_cost = loan_amount * amort_rate
    
    # Calculate other costs
    annual_net_child_cost = calculate_child_cost(row['num_children'])
    annual_income_tax_cost = calculate_taxes(row['gross_yearly_income'], row['tax_rate'])
    annual_housing_cost = calculate_annual_housing_cost(row['housing_type'])
    annual_interest_cost = INTEREST_RATE_FROM_BANK * loan_amount
    annual_cost_of_daily_living = ADULT_COST_OF_DAILY_LIVING_PER_MONTH * MONTHS_OF_A_YEAR
    annual_existing_loans_cost = get_existing_loans_cost(row['customer_id'], existing_loans_lookup)
    
    # Calculate disposable income
    annual_disposable_income = (
        row['gross_yearly_income'] - 
        annual_amortization_cost - 
        annual_net_child_cost - 
        annual_income_tax_cost - 
        annual_housing_cost - 
        annual_interest_cost - 
        annual_cost_of_daily_living - 
        annual_existing_loans_cost
    )
    
    return round(annual_disposable_income)

def calculate_maximum_mortgage_amount(row, existing_loans_lookup):
    """Calculate maximum mortgage amount using correct logic"""
    max_loan_threshold = int(MAX_PERCENTAGE_OF_PROPERTY_VALUATION * row['property_valuation'])
    
    # Check baseline disposable income without any new loan. For efficiency, if they cannot cover the fees today, they cannot cover the fees with a new loan.
    baseline_disposable = calculate_annual_disposable_income(row, 0, existing_loans_lookup)
    if baseline_disposable <= 0:
        return 0
    
    # Use binary search but with much larger steps for efficiency
    low = 0
    high = max_loan_threshold
    
    #Initiating variable
    best_loan = 0
    
    # Binary search with verification
    while high - low > MARGIN_OF_ERROR:
        mid = (low + high) // 2
        disposable_at_mid = calculate_annual_disposable_income(row, mid, existing_loans_lookup)
        
        if disposable_at_mid >= 0:
            best_loan = mid
            low = mid + MARGIN_OF_ERROR
        else:
            high = mid - 1
    
    return int(best_loan)

def process_data(df, existing_loans_lookup, batch_size=10000):
    """Process data in batches"""
    #Using batch processing to avoid memory issues. Print statement to ensure it is working.
    print(f"Processing data in batches of {batch_size}...")
    
    #Initiating list to store the results
    results = []
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        
        #Select rows by integer indexing using pandas iloc
        batch_df = df.iloc[start_idx:end_idx]
        
        #Information for the user to know about the progress
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_df)} records)")
        
        batch_results = []

        #Iterate through the rows in the batch, _ ignores the first column
        for _, row in batch_df.iterrows():
            max_loan = calculate_maximum_mortgage_amount(row, existing_loans_lookup)
            amort_rate = amortization_rate(
                row['property_valuation'], row['requested_loan'], row['gross_yearly_income']
            )
            
            # Calculate all required values for JSON output
            child_cost = calculate_child_cost(row['num_children'])
            taxes = calculate_taxes(row['gross_yearly_income'], row['tax_rate'])
            existing_loans_cost = get_existing_loans_cost(row['customer_id'], existing_loans_lookup)
            disposable_income = calculate_annual_disposable_income(row, row['requested_loan'], existing_loans_lookup)
            
            batch_results.append({
                "application_id": row['application_id'],
                "answer_amortization": float(amort_rate),
                "answer_total_child_cost": int(child_cost),
                "answer_taxes": float(taxes),
                "answer_existing_loans_cost": int(existing_loans_cost),
                "answer_disposable_income": int(disposable_income),
                "answer_max_loan": int(max_loan)
            })
        
        results.extend(batch_results)
    
    return results

# Main execution
if __name__ == "__main__":
    print("Starting optimized mortgage calculation...")
    start_time = time.time()
    
    # Load data
    df, existing_loans_lookup = load_data()
    
    # Process data
    results = process_data(df, existing_loans_lookup, batch_size=10000)
    
    # Save results
    print("Saving results...")
    with open("assignment1_25472.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_time = time.time() - start_time

    #Status update for the user
    print(f"Processing completed in {total_time:.2f} seconds")
    print(f"Processed {len(results)} applications")
    print(f"Average time per application: {total_time/len(results)*1000:.2f} ms")
