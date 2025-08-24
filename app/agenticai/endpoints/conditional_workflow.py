from fastapi import APIRouter, Request
from app.agenticai.conditional_workflow.workflow import ConditionalWorkflow

router = APIRouter()
workflow = ConditionalWorkflow()

@router.post("/agenticai/conditional-fill-form")
async def conditional_fill_form(request: Request):
    request_data = await request.json()
    result = workflow.run(request_data)
    return result
