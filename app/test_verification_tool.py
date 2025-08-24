"""
Test verification tool directly with the payload
"""
import json
from app.llm.tools.form_verification_tool import verify_fee_exemption_form

# Your test payload
test_payload = {
    "message": "Can you please help me fill this water licence application form, please verify if all fields are correct or you need more information",
    "formFields": [
        {
            "data_id": "V1IsEligibleForFeeExemption",
            "fieldLabel": "",
            "fieldType": "radio",
            "fieldValue": "Yes"
        },
        {
            "data_id": "V1IsExistingExemptClient",
            "fieldLabel": "",
            "fieldType": "radio",
            "fieldValue": "Yes"
        },
        {
            "data_id": "V1FeeExemptionClientNumber",
            "fieldLabel": "*Please enter your client number:",
            "fieldType": "text",
            "fieldValue": ""
        },
        {
            "data_id": "V1FeeExemptionCategory",
            "fieldLabel": "*Fee Exemption Category:",
            "fieldType": "select-one",
            "fieldValue": "Federal Government"
        },
        {
            "data_id": "V1FeeExemptionSupportingInfo",
            "fieldLabel": "Please enter any supporting information that will assist in determining your eligibility for a fee exemption.  Please refer to help for details on fee exemption criteria and requirements.",
            "fieldType": "textarea",
            "fieldValue": ""
        }
    ]
}

def test_verification_tool():
    """Test the verification tool directly"""
    try:
        # Convert payload to JSON string
        input_data = json.dumps(test_payload)
        
        # Call the verification tool
        result = verify_fee_exemption_form(input_data)
        
        print("Verification Tool Response:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
    except Exception as e:
        print(f"Error testing verification tool: {str(e)}")

if __name__ == "__main__":
    test_verification_tool()
