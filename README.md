# SupportGPT - Customer Support Agent

A FastAPI-based customer support agent powered by OpenAI that provides intelligent chat responses and conversation summarization capabilities.

## Demo

You can find sample conversations in the `storage/records` directory.  

## Setup

### Prerequisites

- Python 3.13
- OpenAI API key

### Installation

1. **Create a Python 3.13 virtual environment:**
   ```bash
   python3.13 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. **Run the development server:**
   ```bash
   fastapi dev main.py
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Features

- **Chat API**: Intelligent customer support conversations with OpenAI
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Conversation Storage**: Persistent storage of chat conversations
- **Summary Generation**: Automatic conversation summarization
- **Data Collection**: Structured data extraction from conversations

## API Endpoints

- `POST /chat` - Send a message and get AI response
- `POST /chat/summary` - Generate a summary of a conversation
- `GET /docs` - Interactive API documentation


## Development

### Running Tests

```bash
pytest tests/
```

## Key Design Decisions

### Separate `/chat` and `/chat/summary` Endpoints

I decided to split the conversational functionality into two endpoints because they serve different responsibilities:

- **`/chat`** – Handles conversational flows with the user, collecting key information like order number, problem category, and problem description, while also inferring urgency.  
- **`/chat/summary`** – Generates concise summaries of conversations, potentially for internal customer care usage to quickly understand the issue highlights based on the messages history passed.  

This separation also helps manage complexity. The prompts for the agent are already complex, and combining these responsibilities in a single endpoint could interfere with the agent's performance and clarity. Keeping them separate ensures each endpoint can focus on its specific task without introducing errors or unintended behavior.


### Role-Based prompting (system / user / assistant)

All messages sent to the completion API use explicit roles: `system`, `user`, and `assistant`.  

**Benefits:**  
- **Clear context separation** – The `system` role defines the agent’s behavior and rules, `user` provides input, and `assistant` tracks prior responses. This prevents context leakage and keeps the conversation structured.  
- **Improved determinism** – By clearly distinguishing roles, the LLM can follow instructions more reliably, producing consistent outputs.  
- **Production-ready design** – This approach scales well for multi-turn conversations, ensures the agent behaves predictably, and makes debugging or auditing easier since each message’s role is explicit.


### Retry System for OpenAI API Calls

The project includes a robust retry system in the `OpenAIClient` to handle rate limits and transient errors when calling the OpenAI API.  

#### Key Features: 
- **Automatic retries** – Each request is retried up to a configurable number of times (`max_retries`) if rate limits or transient errors occur.  
- **Exponential backoff** – The delay between retries increases exponentially (`base_delay * 2^attempt`) to reduce the likelihood of repeated failures.  
- **Error handling** – Specific handling for rate limit errors (`OpenAIRateLimitError`) and general exception logging, returning clear HTTP error codes when necessary.   

This system ensures the API is **resilient under high load** and "**production-ready**", providing consistent responses even when the OpenAI API throttles requests.


## Future Improvements

**Note**: While SupportGPT is functional for development and testing, additional steps are required to make it production-ready for enterprise use.

Here are some potential enhancements to consider for future development:

### Proper Database Integration

Currently, conversation histories are stored as JSON files. To improve scalability and reliability, we should switch to a proper storage system.

**Options:**
- **AWS S3** – Ideal for long conversation files with high durability.
- **NoSQL Databases** – Flexible alternatives: DynamoDB (scalable, serverless), MongoDB (document-oriented), Redis (in-memory, low-latency).

Choose based on file size, access patterns, and latency requirements.

### Ensure "deterministic" LLM Responses

We need to ensure our LLM responses are as deterministic as possible. Currently, we rely on strong system prompting to guide outputs. But of course, LLM are not deterministic by default so sometimes it does not respect our desired format requested in the prompt. 

In the future, we could leverage the [Completions API `response_format`](https://platform.openai.com/docs/api-reference/runs/createThreadAndRun#runs-createthreadandrun-response_format) to enforce structured outputs. While it doesn’t yet support explicitly specifying a response schema, this feature is likely coming soon, as it’s already available in other OpenAI APIs.


### Leverage Asynchronous API with FastAPI

FastAPI is a high-performance asynchronous Python framework.  

While we are not using it in the API yet, it could:  
- Handle many requests at the same time without blocking  
- Make better use of system resources under high load  
- Improve responsiveness for users when multiple requests happen concurrently


## License

This project is licensed under the MIT License.
