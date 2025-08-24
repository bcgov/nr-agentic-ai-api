"""
Test script for agentic AI form filling system
"""
import asyncio
import json
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agenticai.models import FormFillingRequest, FormField, EXAMPLE_FORM_FIELDS


async def test_agentic_form_filling():
    """Test the agentic form filling system"""
    try:
        from agenticai.workflows import form_filling_workflow
        
        # Test data - water license application form
        test_message = "Can you please help me fill this water licence application form, please verify if all fields are correct or you need more information. I work for the Federal Government and I'm eligible for fee exemption but I don't have a client number yet."
        
        # Convert example form fields to proper format
        form_fields = []
        for field_data in EXAMPLE_FORM_FIELDS:
            form_fields.append(field_data)
        
        user_context = {
            "organization_type": "federal_government",
            "has_client_number": False,
            "application_type": "water_license"
        }
        
        print("🤖 Testing Agentic AI Form Filling System")
        print("=" * 60)
        print(f"📝 User Message: {test_message}")
        print(f"📋 Form Fields: {len(form_fields)} fields")
        print("=" * 60)
        
        # Process the form through the agentic workflow
        result = await form_filling_workflow.process_form(
            message=test_message,
            form_fields=form_fields,
            user_context=user_context
        )
        
        # Display results
        print("🎯 WORKFLOW RESULTS:")
        print("-" * 40)
        print(f"Status: {result['status']}")
        print(f"Workflow State: {result['workflow_state']}")
        print(f"Confidence Score: {result['confidence_score']}")
        
        print("\n✅ FILLED FIELDS:")
        print("-" * 40)
        for field_id, value in result['filled_fields'].items():
            print(f"  {field_id}: {value}")
        
        if result['missing_information']:
            print("\n❌ MISSING INFORMATION:")
            print("-" * 40)
            for missing in result['missing_information']:
                print(f"  - {missing}")
        
        if result['validation_errors']:
            print("\n⚠️ VALIDATION ERRORS:")
            print("-" * 40)
            for error in result['validation_errors']:
                print(f"  - {error['field_id']}: {error['error_message']}")
        
        if result['questions_for_user']:
            print("\n❓ QUESTIONS FOR USER:")
            print("-" * 40)
            for i, question in enumerate(result['questions_for_user'], 1):
                print(f"  {i}. {question}")
        
        if result['suggestions']:
            print("\n💡 SUGGESTIONS:")
            print("-" * 40)
            for i, suggestion in enumerate(result['suggestions'], 1):
                print(f"  {i}. {suggestion}")
        
        print("\n" + "=" * 60)
        print("✨ Test completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"❌ Error testing agentic form filling: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


async def test_form_validation_only():
    """Test form validation functionality only"""
    try:
        from agenticai.tools import analyze_form_structure, validate_field_data
        
        print("\n🔍 Testing Form Validation Only")
        print("=" * 60)
        
        # Test with partially filled form
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
                "field_value": "",  # Missing required field
                "is_required": True
            },
            {
                "data_id": "V1FeeExemptionCategory",
                "field_label": "*Fee Exemption Category:",
                "field_type": "select-one",
                "field_value": "Federal Government",
                "is_required": True
            }
        ]
        
        # Analyze form structure
        analysis = analyze_form_structure.invoke({"form_fields": test_form_fields})
        print(f"📊 Form Analysis:")
        print(f"  - Required fields: {len(analysis.required_fields)}")
        print(f"  - Missing fields: {len(analysis.missing_fields)}")
        print(f"  - Completion: {analysis.completion_percentage:.1f}%")
        
        # Validate field data
        field_data = {field["data_id"]: field["field_value"] for field in test_form_fields}
        validation_errors = validate_field_data.invoke({"field_data": field_data})
        
        print(f"\n🔬 Validation Results:")
        print(f"  - Total errors: {len(validation_errors)}")
        for error in validation_errors:
            print(f"    • {error.field_id}: {error.error_message}")
        
        print("✅ Validation test completed!")
        
    except Exception as e:
        print(f"❌ Error in validation test: {str(e)}")


def test_tools_individually():
    """Test individual tools"""
    try:
        from agenticai.tools import (
            extract_information_from_message,
            generate_suggestions,
            generate_clarifying_questions
        )
        
        print("\n🔧 Testing Individual Tools")
        print("=" * 60)
        
        # Test information extraction
        test_message = "I work for the federal government and need to apply for a water license. I'm eligible for fee exemption."
        test_fields = EXAMPLE_FORM_FIELDS
        
        extracted = extract_information_from_message.invoke({
            "message": test_message,
            "form_fields": test_fields
        })
        
        print("🔍 Information Extraction:")
        for field_id, value in extracted.items():
            print(f"  - {field_id}: {value}")
        
        # Test suggestions
        suggestions = generate_suggestions.invoke({
            "form_fields": test_fields,
            "user_context": {"organization_type": "government"}
        })
        
        print("\n💡 Generated Suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
        # Test question generation
        missing_fields = ["V1FeeExemptionClientNumber"]
        questions = generate_clarifying_questions.invoke({
            "missing_fields": missing_fields,
            "form_fields": test_fields
        })
        
        print("\n❓ Generated Questions:")
        for i, question in enumerate(questions, 1):
            print(f"  {i}. {question}")
        
        print("✅ Tools test completed!")
        
    except Exception as e:
        print(f"❌ Error testing tools: {str(e)}")


async def main():
    """Main test function"""
    print("🚀 Starting Agentic AI Form Filling Tests")
    print("=" * 80)
    
    # Test the full agentic workflow
    await test_agentic_form_filling()
    
    # Test form validation only
    await test_form_validation_only()
    
    # Test individual tools
    test_tools_individually()
    
    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())
