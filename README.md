# AWS-Certified-AI-Practitioner

Introduction to Artificial Intelligence <br>
Introduction to AWS & Cloud Computing <br>
🟡 [Amazon Bedrock and GenAI](#amazon-bedrock-and-genai) <br>
🟡Prompt Engineering <br>
🟡Amazon Q <br>
🟡AI and Machine Learning <br>
🟡AWS Managed AI Services <br>
🟡Amazon SageMaker <br>
🟡Responsible AI, Security, Compliance and Governance <br>
🟡AWS Security Services & More <br>

## Amazon Bedrock and GenAI
### What is Generative AI
- Generative AI (Gen-AI) is a subset of Deep Learning
- Used to generate new data that is similar to the data it was trained on
  - Text
  - Image
  - Audio
  - Code
  - Video…

    <img src = "./images/gen-ai-01.png" width = "300">

### Foundation Model
- To generate data, we must rely on a Foundation Model
- Foundation Models are trained on a wide variety of input data
- The models may cost tens of millions of dollars to train
- Example: GPT-4o is the foundation model behind ChatGPT
- There is a wide selection of Foundation Models from companies:
  - OpenAI
  - Meta (Facebook)
  - Amazon
  - Google
  - Anthropic
- Some foundation models are open-source (free: Meta, Google BERT) and others under a commercial license (OpenAI, Anthropic, etc…)
- Data --- _Training_ ---> Foundation Model --- _Generate New Content_ --->

### Large Language Models (LLM)
- Type of AI designed to generate coherent human-like text
- One notable example: GPT-4 (ChatGPT / Open AI)
- Trained on large corpus of text data
- Usually very big models
  - Billions of parameters
  - Trained on books, articles, websites, other textual data
- Can perform language-related tasks 
  - Translation, Summarization
  - Question answering
  - Content creation

### Generative Language Models
- We usually interact with the LLM by giving a prompt
- Then, the model will leverage all the existing content it has learned from to generate new content
- **Non-deterministic**: the generated text may be different for every user that uses the same prompt

### Generative Language Models
- The LLM generates a list of potential words alongside probabilities
- An algorithm selects a word from that list
- After the rain, the streets were
  - wet (0.40)
  - flooded (0.25)
  - slippery (0.15)
  - empty (0.10)
  - muddy (0.05)
  - clean (0.03)
  - blocked (0.02)
- After the rain, the streets were flooded
  - and (0.30)
  - with (0.20)
  - but (0.15)
  - from (0.12)
  - until (0.10)
  - because (0.08)
  - -- (0.05)

### Generative Images from Text Prompts
**Prompt**: Generate a blue sky with white clouds and the word "Hello" written in the sky
### Generative Images from Images
**Prompt**: Transform this image in Japanese anime style
### Generative Text from Images
**Prompt**: Describe how many apples you see in the picture

### Diffusion Models (ex: Stable Diffusion)
<img src = "./images/diffusion-model.png" width = "300">

## Amazon Bedrock
- Build Generative AI (Gen-AI) applications on AWS
- Fully-managed service, no servers for you to manage
- Keep control of your data used to train the model
- Pay-per-use pricing model
- Unified APIs
- Leverage a wide array of foundation models
- Out-of-the box features: RAG, LLM Agents…
- Security, Privacy, Governance and Responsible AI features
  <img src = "./images/amazon-bedrock-01.png" width = "300">

### Foundation Models
- Access to a wide range of Foundation Models (FM):
  - AI21labs
  - cohere
  - stability.ai
  - amazon
  - ANTHROPIC
  - Meta
  - MISTRAL AI
- Amazon Bedrock makes a copy of the FM, available only to you, which you can further fine-tune with your own data
- None of your data is used to train the FM

### Sample Architecture
<img src = "./images/amazon-bedrock-02.png" width = "300">

### Base Foundation Model
- How to choose?
  - Model types, performance requirements, capabilities, constraints, compliance 
  - Level of customization, model size, inference options, licensing agreements, context windows, latency 
  - Multimodal models (varied types of input and outputs)
- What's Amazon Titan?
  - High-performing Foundation Models from AWS
  - Image, text, multimodal model choices via a fully-managed APIs
  - Can be customized with your own data
- Smaller models are most cost-effective

### Example Comparison Between Different Models
| | **Amazon Titan** <br> (Titan Test Express)     | **Llama** <br> (Llama-2 70b-chat)    | **Claude** <br> (Claude 2.1)                  | **Stable Diffusion** <br> (SDXL 1.0)     |
|-|------------------------------------------------|--------------------------------------|-----------------------------------------------|------------------------------------------|
| Max Tokens (=max context window) | 8K Tokens                                      | 4K Tokens                            | 200K Tokens                                   | 77-Tokens/Prompt                         |
| Features | High-performance text model, +100 languages    | Large-scale tasks, dialogue, English | High-capacity text generation, multi-language | Image generation                         |
| Use Cases | Content creation, classification, education... | Text generation, customer service    | Analysis, forecasting, document copmarison... | Image creation for advertising, media... |
| Pricing (1K Tokens) | Input: $0.0008 <br> Output: $0.0016            | Input: $0.0019 <br> Output: $0.0025  | Input: $0.0008 <br> Output: $0.0024           | $0.04 - 0.08 / image                     |

### Fine-Tuning a Model
- Adapt a copy of a foundation model with your own data
- Fine-tuning will change the weights of the base foundation model
- Training data must:
- Adhere to a specific format
  - Be stored in Amazon S3
  - You must use “Provisioned Throughput” to use a fine-tuned model
- Note: not all models can be fine-tuned

    <img src = "./images/fine--tuning-a-model.png" width = "300">

### Instruction Based Fine Tuning
- Improves the performance of a pre-trained FM on domain specific tasks
- = further trained on a particular field or area of knowledge
- Instruction-based fine-tuning uses labeled examples that are prompt-response pairs
 <img src = "./images/labeled-data.png" width = "300">

### Continued Pre-Training
- Provide unlabeled data training of an FM to continue the
- Also called domain-adaptation fine-tuning, to make a model expert in a specific domain
- For example: feeding the entire AWS documentation to a model to make it an expert on AWS
- Good to feed industry-specific terminology into a model (acronyms, etc…)
- Can continue to train the model as more data becomes available
<img src = "./images/continue-pre--training.png" width = "300">

### Single-Turn Messaging
- Part of instruction-based fine-tuning
- system (optional) : context for the conversation.
- messages : An array of message objects, each containing:
  - role : Either user or assistant
  - content : The text content of the message
<img src = "./images/single--turn-messaging.png" width = "300">

### Multi-Turn Messaging
- To provide instruction based fine-tuning for a conversation (vs Single Turn Messaging)
- Chatbots = multi-turn environment
- You must alternate between “user” and “assistant” roles
<img src = "./images/multi--turn-messaging.png" width = "300">

### Fine-Tuning: Good to Know
- Re-training an FM requires a higher budget
- Instruction-based fine-tuning is usually cheaper as computations are less intense and the amount of data required usually less
- It also requires experienced ML engineers to perform the task
- You must prepare the data, do the fine-tuning, evaluate the model
- Running a fine-tuned model is also more expensive (provisioned throughput)

### Note: Transfer Learning
- **Transfer Learning** – the broader concept of re-using a pre-trained model to adapt it to a new related task
  - Widely used for image classification
  - And for NLP (models like BERT and GPT)
- Can appear in the exam as a general ML concept
- Fine-tuning is a specific kind of transfer learning
- Claude3 (Pre-trained model) ➡️ Transfer Learning ➡️ Model adapter to a new task

### Fine-Tuning - Use Cases
- A chatbot designed with a particular persona or tone, or geared towards a specific purpose (e.g., assisting customers, crafting advertisements)
- Training using more up-to-date information than what the language model previously accessed
- Training with exclusive data (e.g., your historical emails or messages, records from customer service interactions)
- Targeted use cases (categorization, assessing accuracy)

### Amazon Bedrock - Evaluating a Model (Automatic Evaluation)
- Evaluate a model for quality control
- Built-in task types:
  - Text summarization
  - question and answer
  - text classification
  - open-ended text generation…
- Bring your own prompt dataset or use built-in curated prompt datasets
- Scores are calculated automatically
- Model scores are calculated using various statistical methods (e.g. BERTScore, F1…)
<img src = "./images/automatic-evaluation.png" width = "300">

### Note on Benchmark Datasets
- Curated collections of data designed specifically at evaluating the performance of language models
- Wide range of topics, complexities, linguistic phenomena
- Helpful to measure: accuracy, speed and efficiency, scalability
- Some benchmarks datasets allow you to very quickly detect any kind of bias and potential discrimination against a group of people
- You can also create your own benchmark dataset that is specific to your business

### Amazon Bedrock - Evaluating a Model (Human Evaluation)
- Choose a work team to evaluate
    - Employees of your company
    - Subject-Matter Experts (SMEs)
- Define metrics and how to evaluate
  - Thumbs up/down, ranking…
  - Choose from Built-in task types (same as Automatic) or add a custom task
  <img src = "./images/human-evaluation.png" width = "300">

### Automated Metrics to Evaluate an FM
- **ROGUE**: Recall-Oriented Understudy for Gisting Evaluation
  -  Evaluating automatic summarization and machine translation systems
  - ROUGE-N – measure the number of matching n-grams between reference and generated text
  - ROUGE-L – longest common subsequence between reference and generated text
- **BLEU**: Bilingual Evaluation Understudy
  - Evaluate the quality of generated text, especially for translations
  - Considers both precision and penalizes too much brevity
  - Looks at a combination of n-grams (1, 2, 3, 4)
- **BERTScore**:
  - Semantic similarity between generated text
  - Uses pre-trained BERT models (Bidirectional Encoder Representations from Transformers) to compare the contextualized embeddings of both texts and computes the cosine similarity between them.
  - Capable of capturing more nuance between the texts
- **Perplexity**: how well the model predicts the next token (lower is better)

### Automated Model Evaluation
<img src = "./images/automated-model-evaluation.png" width = "300">

### Business Metrics to Evaluate a Model On
- **User Satisfaction**: gather users’ feedbacks and assess their satisfaction with the
model responses (e.g., user satisfaction for an ecommerce platform)
- **Average Revenue Per User (ARPU)**: average revenue per user attributed to
the Gen-AI app (e.g., monitor ecommerce user base revenue)
- **Cross-Domain Performance**: measure the model’s ability to perform cross
different domains tasks (e.g., monitor multi-domain ecommerce platform)
- **Conversion Rate**: generate recommended desired outcomes such as purchases
(e.g., optimizing ecommerce platform for higher conversion rate)
- **Efficiency**: evaluate the model’s efficiency in computation, resource utilization…
(e.g., improve production line efficiency)

### Amazon Bedrock - RAG & Knowledge Base
- RAG = Retrieval-Augmented Generation
- Allows a Foundation Model to reference a data source outside its training data
- Bedrock takes care of creating Vector Embeddings in the database of your choice based on your data
- Use where real-time data is needed to be fed into the Foundation Model

    <img src = "./images/rag&knowledgebase.png" width = "300">

### Amazon Bedrock - RAG in Action
<img src = "./images/rag-in-action.png" width = "300">

### Amazon Bedrock - RAG Vector Databases
<img src = "./images/rag-vector-databases.png" width = "300">

### RAG Vector Databases - Types
- **Amazon OpenSearch Service**: search & analytics database real time similarity queries, store millions of vector embeddings scalable index management, and fast nearest-neighbor (kNN) search capability
- **Amazon DocumentDB**: [with MongoDB compatibility] - NoSQL database real time similarity queries, store millions of vector embeddings
- **Amazon Aurora**: relational database, proprietary on AWS
- **Amazon RDS for PostgreSQL**: relational database, open-source
- **Amazon Neptune**: graph database

### RAG Data Sources
- Amazon S3
- Confluence
- Microsoft Sharepoint
- Web pages (website, social media feed, etc)
- _More added over time_

### Amazon Bedrock - RAG Use Cases
- **Customer Service Chatbot**
    - **Knowledge Base** – products, features, specifications, troubleshooting guides, and FAQs
    - **RAG application** – chatbot that can answer customer queries
- **Legal Research and Analysis**
  - **Knowledge Base** – laws, regulations, case precedents, legal opinions, and expert analysis
  - **RAG Application** – chatbot that can provide relevant information for specific legal queries
- **Healthcare Question-Answering**
  - **Knowledge base** – diseases, treatments, clinical guidelines, research papers, patients…
  - **RAG application** – chatbot that can answer complex medical queries

### GenAI Concepts - Tokenization
- **Tokenization**: conver ting raw text into a sequence of tokens
    - Word-based tokenization: text is split into individual words
    - Subword tokenization: some words can be split too (helpful for long words…)
- Can experiment at: https://platform.openai.com/tokenizer

### GenAI Concepts - Context Window
- The number of tokens an LLM can consider when generating text
- The larger the context window, the more information and coherence
- Large context windows require more memory and processing power
- First factor to look at when considering a model

### GenAI Concepts - Embeddings
- Create vectors (array of numerical values) out of text, images or audio
- Vectors have a high dimensionality to capture many features for one input token, such as semantic meaning, syntactic role, sentiment
- Embedding models can power search applications
<img src = "./images/embeddings.png" width = "300">

#### Words that have a Semantic Relationship have similar Embeddings
<img src = "./images/similar-embeddings.png" width = "300">

### Amazon Bedrock - Guardrails
- Control the interaction between users and Foundation Models (FMs)
- Filter undesirable and harmful content
- Remove Personally Identifiable Information (PII)
- Enhanced privacy
- Reduce hallucinations
- Ability to create multiple Guardrails and monitor and analyze user inputs that can violate the Guardrails
<img src = "./images/bedrock-guardrails.png" width = "300">

### Amazon Bedrock - Agents
- Manage and carry out various multi-step tasks related to infrastructure provisioning, application deployment, and operational activities
- Task coordination: perform tasks in the correct order and ensure information is passed correctly between tasks
- Agents are configured to perform specific pre-defined action groups
- Integrate with other systems, services, databases and API to exchange data or initiate actions
- Leverage RAG to retrieve information when necessary

### Bedrock Agent Setup
<img src = "./images/bedrock-agent-setup.png" width = "300">

### Agent Diagram
<img src = "./images/agent-diagram.png" width = "300">

### Amazon Bedrock & CloudWatch
- **Model Invocation Logging**
  - Send logs of all invocations to Amazon CloudWatch and S3
    - Can include text, images and embeddings
      - Analyze further and build alerting thanks to CloudWatch Logs Insights
- **CloudWatch Metrics**
  - Published metrics from Bedrock to CloudWatch
    - Including ContentFilteredCount, which helps to see if Guardrails are functioning
  - Can build CloudWatch Alarms on top of Metrics
<img src = "./images/bedrock-cloudwatch.png" width = "300">

### Amazon Bedrock - Other Features
- **Bedrock Studio** – give access to Amazon Bedrock to your team so they can easily create AI-powered applications
- **Watermark detection** – check if an image was generated by Amazon Titan Generator

### Amazon Bedrock - Pricing
- **On-Demand**
  - Pay-as-you-go (no commitment)
  - Text Models – charged for every input/output token processed
  - Embedding Models – charged for every input token processed
  - Image Models – charged for every image generated
  - Works with Base Models only
- **Batch**:
  - Multiple predictions at a time (output is a single file in Amazon S3)
  - Can provide discounts of up to 50%
- **Provisioned Throughput**:
  - Purchase Model units for a certain time (1 month, 6 months…)
  - Throughput – max. number of input/output tokens processed per minute
  - Works with Base, Fine-tuned, and Custom Models

### Model Improvement Techniques Cost Order
1. **Prompt Engineering**: 
    - No model training needed (no additional computation or fine-tuning)
2. **Retrieval Augmented Generation (RAG)**:
   - Uses external knowledge (FM doesn’t need to ”know everything”, less complex)
   - No FM changes (no additional computation or fine-tuning)
3. **Instruction-based Fine-tuning**:
   - FM is fine-tuned with specific instructions (requires additional computation
4. **Domain Adaption Fine-tuning**:
   - Model is trained on a domain-specific dataset (requires intensive computation)

### Bedrock - Cost Savings
- **On-Demand** – great for unpredictable workloads, no long-term commitment 
- **Batch** – provides up to 50% discounts
- **Provisioned Throughput** – (usually) not a cost-saving measure, great to “reserve” capacity
- **Temperature, Top K, Top P** – no impact on pricing
- **Model size** – usually a smaller model will be cheaper (varies based on providers)
- **Number of Input and Output Tokens** – main driver of cost

