"""
Test script for water agent form verification
"""
import json
from app.llm.agents.water_agent import water_executor

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

def test_water_agent():
    """Test the water agent with the payload"""
    try:
        # Convert payload to JSON string as the agent expects
        input_text = json.dumps(test_payload)
        
        # Call the water agent
        result = water_executor.invoke({
            "input": input_text
        })
        
        print("Water Agent Response:")
        print("=" * 50)
        print(result.get("output", "No output"))
        print("=" * 50)
        
    except Exception as e:
        print(f"Error testing water agent: {str(e)}")

if __name__ == "__main__":
    test_water_agent()
