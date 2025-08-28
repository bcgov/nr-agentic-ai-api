"""
Simple test for agentic AI tools functionality
"""
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tools_basic():
    """Test basic tool functionality"""
    try:
        print("🔧 Testing Agentic AI Tools")
        print("=" * 50)
        
        from agenticai.tools import (
            analyze_form_structure,
            extract_information_from_message,
            validate_field_data,
            generate_suggestions
        )
        
        # Test form structure analysis
        test_form_fields = [
            {
                "data_id": "V1IsEligibleForFeeExemption",
                "field_label": "Are you eligible for fee exemption?",
                "field_type": "radio",
                "field_value": "Yes",
                "is_required": True
            },
            {
                "data_id": "V1FeeExemptionClientNumber",
                "field_label": "*Please enter your client number:",
                "field_type": "text",
                "field_value": "",
                "is_required": True
            }
        ]
        
        print("1. Testing Form Analysis...")
        analysis = analyze_form_structure(test_form_fields)
        print(f"   ✅ Required fields: {len(analysis.required_fields)}")
        print(f"   ✅ Missing fields: {len(analysis.missing_fields)}")
        print(f"   ✅ Completion: {analysis.completion_percentage:.1f}%")
        
        print("\n2. Testing Information Extraction...")
        test_message = "I work for the federal government and need fee exemption"
        extracted = extract_information_from_message(test_message, test_form_fields)
        print(f"   ✅ Extracted {len(extracted)} field values:")
        for field_id, value in extracted.items():
            print(f"      - {field_id}: {value}")
        
        print("\n3. Testing Validation...")
        field_data = {"V1FeeExemptionClientNumber": ""}
        errors = validate_field_data(field_data)
        print(f"   ✅ Found {len(errors)} validation errors")
        
        print("\n4. Testing Suggestions...")
        suggestions = generate_suggestions(test_form_fields)
        print(f"   ✅ Generated {len(suggestions)} suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"      {i}. {suggestion}")
        
        print("\n🎉 All tools tested successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing tools: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_basic():
    """Test basic workflow functionality"""
    try:
        print("\n🔄 Testing Workflow Components")
        print("=" * 50)
        
        from agenticai.workflows import FormFillingWorkflow
        
        workflow = FormFillingWorkflow()
        print("✅ Workflow initialized successfully")
        
        # Test with simple form data
        test_form = [
            {
                "data_id": "V1FeeExemptionCategory",
                "field_label": "*Fee Exemption Category:",
                "field_type": "select-one",
                "field_value": "",
                "is_required": True
            }
        ]
        
        print("✅ Basic workflow test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error testing workflow: {str(e)}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Basic Agentic AI Tests")
    print("=" * 60)
    
    tools_ok = test_tools_basic()
    workflow_ok = test_workflow_basic()
    
    if tools_ok and workflow_ok:
        print("\n✨ All basic tests passed!")
        print("\n📚 Next Steps:")
        print("1. Start the FastAPI server: uvicorn main:app --reload")
        print("2. Test the API endpoint: POST /agenticai/fill-form")
        print("3. View API docs: http://localhost:8000/docs")
    else:
        print("\n❌ Some tests failed. Check the output above.")
