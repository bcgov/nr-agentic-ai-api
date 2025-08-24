"""
Direct test of the water agent with verbose output
"""
from app.llm.agents.water_agent import water_executor

def test_with_your_exact_payload():
    """Test with the exact payload format you're using"""
    
    # Your exact message
    user_message = """Can you please help me fill this water licence application form, please verify if all fields are correct or you need more information

Form data: {"formFields": [{"data_id": "V1IsEligibleForFeeExemption", "fieldValue": "Yes"}, {"data_id": "V1IsExistingExemptClient", "fieldValue": "Yes"}, {"data_id": "V1FeeExemptionClientNumber", "fieldValue": ""}, {"data_id": "V1FeeExemptionCategory", "fieldValue": "Federal Government"}, {"data_id": "V1FeeExemptionSupportingInfo", "fieldValue": ""}]}"""
    
    try:
        print("Testing with message:", user_message[:100] + "...")
        print("=" * 50)
        
        result = water_executor.invoke({
            "input": user_message
        })
        
        print("Final Output:")
        print(result.get("output", "No output"))
        
        if "intermediate_steps" in result:
            print("\nAgent Steps:")
            for i, step in enumerate(result["intermediate_steps"]):
                print(f"Step {i+1}:")
                print(f"  Tool: {step[0].tool}")
                print(f"  Input: {step[0].tool_input[:100]}...")
                print(f"  Output: {step[1][:200]}...")
                print()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_your_exact_payload()
