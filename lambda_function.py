import boto3
import random
import json

# DynamoDB connection
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("CloudFacts")

# Bedrock client
bedrock = boto3.client("bedrock-runtime")

# Allowed frontend origins for CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # local dev
    "https://production.d3bg0cbuq9nbk.amplifyapp.com"  # deployed Amplify frontend
]

def invoke_bedrock(model_id, prompt, max_tokens=50, temperature=0.8):
    """Call Bedrock model with error handling"""
    body = {
        "inputText": prompt,
        "maxTokens": max_tokens,
        "temperature": temperature
    }
    try:
        resp = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json"
        )
        result = json.loads(resp["body"].read())
        return result.get("outputText", "").strip()
    except Exception as e:
        print(f"Bedrock error with {model_id}: {e}")
        return None

def lambda_handler(event, context):
    # Determine the request origin
    request_origin = event.get("headers", {}).get("origin") or event.get("headers", {}).get("Origin")
    allow_origin = request_origin if request_origin in ALLOWED_ORIGINS else "null"

    # Handle OPTIONS preflight request (CORS)
    if event.get("httpMethod") == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": ""
        }

    # Fetch all facts from DynamoDB
    response = table.scan()
    items = response.get("Items", [])
    if not items:
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"fact": "No facts available in DynamoDB."})
        }

    # Pick a random fact
    fact = random.choice(items)["FactText"]

    # Create prompt for witty rewrite
    prompt = f"""
You are a witty, funny assistant.
Rewrite the following cloud computing fact to be short, fun, and engaging, in 1-2 sentences maximum.
Use humor, wordplay, or clever comparisons if possible.

Fact: {fact}

Rewritten Fact:
"""

    # Try Nova Pro first
    witty_fact = invoke_bedrock("amazon.nova-pro-v1:0", prompt)

    # Fallback to Nova Lite
    if not witty_fact:
        print("Falling back to Nova Lite due to Nova Pro throttling/error")
        witty_fact = invoke_bedrock("amazon.nova-lite-v1:0", prompt)

    # If still empty, return original fact
    if not witty_fact:
        witty_fact = fact

    # Return response with CORS headers
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
        "body": json.dumps({"fact": witty_fact})
    }