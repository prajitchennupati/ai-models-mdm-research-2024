import pandas as pd

# Sample DataFrame
data = {
    'phone_numbers': ['1234567890', '9876543210', '123-456-7890', '456.789.0123', 'not_a_number']
}

df = pd.DataFrame(data)

def format_phone_number(phone):
    # Remove all non-numeric characters
    digits = ''.join(filter(str.isdigit, phone))
    
    # Check if we have exactly 10 digits
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    else:
        return "Invalid number"

# Apply the formatting function to the DataFrame
df['formatted_phone_numbers'] = df['phone_numbers'].apply(format_phone_number)

print(df)
