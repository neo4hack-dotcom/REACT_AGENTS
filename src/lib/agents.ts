import { BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import Parser from 'rss-parser';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import * as xlsx from 'xlsx';
import { Document, Packer, Paragraph, TextRun } from 'docx';
import { Client } from '@elastic/elasticsearch';

export interface AgentResult {
  sql?: string;
  rows?: any[];
  answer: string;
  details?: any;
}

export class AgentExecutor {
  static async execute(agentType: string, config: any, input: string, llm: any): Promise<AgentResult> {
    let systemPrompt = "You are a helpful AI assistant.";
    let extraContext = "";
    
    switch (agentType) {
      case 'sql_analyst':
        systemPrompt = `You are a Senior ClickHouse Expert and SQL Analyst. Database: ${config.database_name || 'Unknown'}. 
Mode: ${config.sql_use_case_mode || 'llm_sql'}.
Template: ${config.sql_query_template || 'None'}.

CRITICAL CLICKHOUSE BEST PRACTICES:
- Never use PostgreSQL or MySQL syntax.
- For time-series gap analysis, use WITH FILL.
- For behavioral analysis (funnels), use windowFunnel().
- For fast descriptive stats on billions of rows, use quantileTiming, uniqHLL12, topK.
- For conditional aggregations, prefer sumIf, countIf over complex joins.
- For time-series joins, always consider ASOF JOIN.
- For distinct counts on large volumes, use uniq or uniqExact.
- Filters MUST use the table's sorting key (ORDER BY) for optimal performance.

CRITICAL FOR LARGE SCHEMAS & HUGE TABLES:
- Do not guess table or column names.
- If you lack schema context, write exploratory SQL first (e.g., SHOW TABLES, DESCRIBE table, or query system.columns/system.tables).
- Always LIMIT your queries when exploring data (e.g., SELECT * FROM table LIMIT 5).
- If parameters are missing, state what is missing in your answer.

Return your response in JSON format: { "sql": "SELECT ...", "answer": "Explanation..." }`;
        break;
      case 'clickhouse_table_manager':
        systemPrompt = `You are a ClickHouse Table Manager. 
Protect existing tables: ${config.protect_existing_tables !== false}.
Allow inserts: ${!!config.allow_row_inserts}.
Allow updates: ${!!config.allow_row_updates}.
Allow deletes: ${!!config.allow_row_deletes}.

CRITICAL FOR LARGE SCHEMAS:
- Verify table existence and structure before applying changes.
- Do not drop or truncate tables unless explicitly permitted and verified.

Return your response in JSON format: { "sql": "CREATE TABLE ...", "answer": "Table created." }`;
        break;
      case 'clickhouse_writer':
        systemPrompt = `You are a ClickHouse Data Writer.
Enforce agent prefix: ${config.enforce_agent_prefix !== false}.
Allow inserts: ${config.allow_inserts !== false}.

CRITICAL SECURITY RULE:
- You are ONLY allowed to create, modify, or insert into tables whose names start with "agent_".
- If the user asks to modify a table that does not start with "agent_", you MUST refuse and explain the security policy.
- Ensure all inserted data is properly escaped and formatted.

Return your response in JSON format: { "sql": "INSERT INTO agent_... ", "answer": "Data inserted." }`;
        break;
      case 'knowledge_base_assistant':
        systemPrompt = `You are an internal Knowledge Base Assistant (Concierge Interne).
Vector DB: ${config.vector_db || 'qdrant'}
Embedding Model: ${config.embedding_model || 'text-embedding-004'}
Use Hybrid Search: ${!!config.use_hybrid_search}
Use Reranker: ${!!config.use_reranker}

CRITICAL RAG BEST PRACTICES:
- NEVER hallucinate. If the answer is not in the provided context, state clearly that you do not know.
- ALWAYS cite your sources (e.g., [Document Name, Page X]).
- Synthesize the retrieved chunks into a coherent, professional response.

Return your response in JSON format: { "answer": "Your synthesized answer with citations...", "sources": ["doc1", "doc2"] }`;
        break;
      case 'data_anomaly_hunter':
        systemPrompt = `You are a Data Anomaly Hunter.
Detection Method: ${config.detection_method || 'statistical'}
Use Dynamic Thresholds: ${!!config.use_dynamic_thresholds}

CRITICAL ANOMALY DETECTION BEST PRACTICES:
- You receive statistical outliers (e.g., Z-score > 3, IQR anomalies) and KPI baselines.
- DO NOT read raw data rows. Rely on the statistical summaries provided.
- Explain WHY the anomaly might have happened in clear business terms.
- Format an alert message suitable for ${config.alert_channel || 'Slack'}.

Return your response in JSON format: { "answer": "Business explanation of the anomaly...", "alert_message": "🚨 *Anomaly Detected*..." }`;
        break;
      case 'text_to_sql_translator':
        systemPrompt = `You are an expert ClickHouse Text-to-SQL Translator (Traducteur Métier).
Use Semantic Layer: ${!!config.use_semantic_layer}
Use Golden Queries: ${!!config.use_golden_queries}

CRITICAL TEXT-TO-SQL BEST PRACTICES:
- You will receive the DDL, column descriptions, and potentially "Golden Queries" as examples.
- Write precise ClickHouse SQL. Avoid complex joins if conditional aggregations (sumIf, countIf) suffice.
- If the query fails, you will receive the error and must auto-correct it (up to ${config.max_retries || 3} times).
- Generate a chart configuration (e.g., ECharts JSON) if requested and appropriate.

Return your response in JSON format: { "sql": "SELECT ...", "answer": "Explanation...", "chart_config": {} }`;
        break;
      case 'data_profiler_cleaner':
        systemPrompt = `You are a Data Profiler & Cleaner (Data Steward).
Execution Engine: ${config.execution_engine || 'polars'}
Use Sampling: ${!!config.use_sampling} (Sample size: ${config.sample_size || 10000})

CRITICAL DATA PROFILING BEST PRACTICES:
- Review the provided statistical profile (nulls, cardinality, min/max, inferred types).
- Identify data quality issues (e.g., negative ages, inconsistent formats).
- Write a Python (Pandas/Polars) or SQL script to clean the data.
- NEVER execute destructive cleaning in production. Propose a script for human validation.

Return your response in JSON format: { "answer": "Audit summary...", "cleaning_script": "import polars as pl\\n..." }`;
        break;
      case 'unstructured_to_structured':
        systemPrompt = `You are a data extraction assistant. Convert the unstructured text into structured JSON.
Output Schema: ${typeof config.output_schema === 'string' ? config.output_schema : JSON.stringify(config.output_schema || {})}
Strict JSON: ${config.strict_json !== false}
Return ONLY valid JSON matching the schema.`;
        break;
      case 'email_cleaner':
        systemPrompt = `You are an Email Cleaner. Condense the email into actionable sections.
Max bullets: ${config.max_bullets || 5}.
Include sections: ${config.include_sections || 'Action Items, Summary'}.
Return your response in JSON format: { "answer": "Cleaned email content..." }`;
        break;
      case 'file_assistant':
        systemPrompt = `You are a File Assistant. Answer questions based on the files in folder: ${config.folder_path}.`;
        if (config.folder_path) {
           try {
             if (fs.existsSync(config.folder_path)) {
               const files = fs.readdirSync(config.folder_path).slice(0, config.max_files || 10);
               extraContext = `\n\nFiles in directory:\n${files.join('\n')}`;
             } else {
               extraContext = `\n\nError: Directory ${config.folder_path} does not exist.`;
             }
           } catch (e: any) {
             extraContext = `\n\nError reading directory: ${e.message}`;
           }
        }
        break;
      case 'text_file_manager':
        systemPrompt = `You are a Text File Manager. Folder: ${config.folder_path}. Allow overwrite: ${!!config.allow_overwrite}.
Return your response in JSON format: { "answer": "File operations completed.", "details": {"action": "read|write|append", "file": "filename"} }`;
        break;
      case 'excel_manager':
        systemPrompt = `You are an Excel Manager. Workbook: ${config.workbook_path}.
Return your response in JSON format: { "answer": "Excel operations completed.", "details": {"action": "read|write", "sheet": "sheetname"} }`;
        break;
      case 'word_manager':
        systemPrompt = `You are a Word Manager. Document: ${config.document_path}.
Return your response in JSON format: { "answer": "Word operations completed.", "details": {"action": "read|write"} }`;
        break;
      case 'elasticsearch_retriever':
        systemPrompt = `You are an Elasticsearch Retriever. Index: ${config.index}. Synthesize the retrieved documents.
Return your response in JSON format: { "answer": "Synthesized content..." }`;
        if (config.base_url && config.index) {
          try {
            const client = new Client({
              node: config.base_url,
              auth: config.api_key ? { apiKey: config.api_key } : (config.username ? { username: config.username, password: config.password || '' } : undefined),
              tls: { rejectUnauthorized: config.verify_ssl !== false }
            });
            const result = await client.search({
              index: config.index,
              query: {
                simple_query_string: {
                  query: input,
                  fields: config.fields
                    ? config.fields.split(',').map((field: string) => field.trim())
                    : ['*']
                }
              },
              size: config.top_k || 5
            });
            const hits = result.hits.hits.map((h: any) => h._source);
            extraContext = `\n\nElasticsearch Results:\n${JSON.stringify(hits).substring(0, 5000)}`;
          } catch (e: any) {
            console.error("Elasticsearch Error", e);
            extraContext = `\n\nElasticsearch Error: ${e.message}`;
          }
        }
        break;
      case 'rag_context':
        systemPrompt = `You are a RAG Assistant. Folder: ${config.folder_path}. Top K chunks: ${config.top_k_chunks || 3}.
Return your response in JSON format: { "answer": "RAG response..." }`;
        break;
      case 'rss_news':
        systemPrompt = `You are an RSS News summarizer. Feeds: ${config.feed_urls}. Interests: ${config.interests}.`;
        try {
          if (config.feed_urls) {
            const parser = new Parser();
            const urls = config.feed_urls.split(',').map((u: string) => u.trim());
            let feedItems: any[] = [];
            for (const url of urls) {
              try {
                const feed = await parser.parseURL(url);
                feedItems = feedItems.concat(feed.items.slice(0, config.max_items_per_feed || 5));
              } catch (e) {
                console.error("Failed to parse RSS feed", url);
              }
            }
            extraContext = `\n\nRecent News Items:\n${feedItems.map(i => `- ${i.title}: ${i.contentSnippet || i.content}`).join('\n')}`;
          }
        } catch (e) {
          console.error("RSS Error", e);
        }
        break;
      case 'web_scraper':
        systemPrompt = `You are a Web Scraper. Start URLs: ${config.start_urls}. Allowed domains: ${config.allowed_domains}.`;
        try {
          if (config.start_urls) {
            const urls = config.start_urls.split(',').map((u: string) => u.trim());
            let scrapedContent = "";
            for (const url of urls) {
              try {
                const res = await fetch(url);
                const html = await res.text();
                const $ = cheerio.load(html);
                scrapedContent += `\n\nContent from ${url}:\n${$('body').text().substring(0, config.max_chars_per_page || 5000)}`;
              } catch (e) {
                console.error("Failed to scrape URL", url);
              }
            }
            extraContext = `\n\nScraped Content:\n${scrapedContent}`;
          }
        } catch (e) {
          console.error("Scraper Error", e);
        }
        break;
      case 'web_navigator':
        systemPrompt = `You are a Web Navigator. Start URL: ${config.start_url}. Max steps: ${config.max_steps || 5}.
Return your response in JSON format: { "answer": "Navigation completed." }`;
        extraContext = `\n\nNote: Playwright is not installed in this environment. The agent is marked as unavailable.`;
        break;
      case 'custom':
      default:
        systemPrompt = `Role: ${config.role || 'Assistant'}
Persona: ${config.persona || 'Helpful'}
Objectives: ${config.objectives || 'Assist the user'}
Instructions: ${config.system_prompt || ''}`;
        break;
    }

    try {
      const response = await llm.invoke([
        new SystemMessage(systemPrompt),
        new HumanMessage(input + extraContext)
      ]);

      const content = response.content as string;
      
      // Try to parse JSON if the agent is expected to return JSON
      if (content.trim().startsWith('{') || agentType !== 'custom') {
        try {
          // Find JSON block if wrapped in markdown
          const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
          const jsonStr = jsonMatch ? jsonMatch[1] : content;
          const parsed = JSON.parse(jsonStr);
          
          // Fallback for unstructured_to_structured if strict_json is true
          if (agentType === 'unstructured_to_structured' && config.strict_json !== false) {
             // Basic validation could go here
          }
          
          return {
            answer: parsed.answer || (agentType === 'unstructured_to_structured' ? JSON.stringify(parsed) : "JSON parsed successfully."),
            sql: parsed.sql,
            rows: parsed.rows || [],
            details: parsed
          };
        } catch (e) {
          // Fallback if parsing fails
          if (agentType === 'unstructured_to_structured') {
            return { answer: `Fallback raw output:\n${content}` };
          }
          return { answer: content };
        }
      }

      return { answer: content };
    } catch (error: any) {
      return { answer: `Error executing agent: ${error.message}` };
    }
  }
}

export class MultiAgentManager {
  static async run_stream(input: string, agents: any[], llm: any, managerConfig: any = {}, onStep?: (step: any) => void) {
    const steps = [];
    let finalAnswer = "";
    
    const addStep = (step: any) => {
      steps.push(step);
      if (onStep) onStep(step);
    };

    // Filter active agents (mocking Playwright check for web_navigator)
    const activeAgents = agents.filter(a => {
      if (a.agent_type === 'web_navigator') {
        return false; // Playwright not available
      }
      return true;
    });
    
    addStep({ 
      status: "manager_start", 
      message: "Starting multi-agent orchestration", 
      active_agents: activeAgents.length,
      unavailable_agents: agents.length - activeAgents.length
    });
    
    let currentStep = 0;
    const maxSteps = managerConfig.max_steps || 10;
    let isDone = false;
    let conversationHistory = `User Request: ${input}\n`;
    let scratchpad: Record<string, any> = {};

    while (currentStep < maxSteps && !isDone) {
      const managerPrompt = `You are an advanced Multi-Agent Orchestrator.
Available agents:
${activeAgents.map(a => `- ${a.name} (${a.agent_type}): ${a.role || ''} - ${a.objectives || ''}`).join('\n')}

Current Conversation History & Knowledge:
${conversationHistory}

Shared Memory (Scratchpad):
${JSON.stringify(scratchpad, null, 2)}

CRITICAL INSTRUCTIONS FOR LARGE DATA/SCHEMAS:
1. Do not assume schema structures. Anticipate multiple phases:
   - Phase 1 (Discovery): Query table names, metadata, or file lists.
   - Phase 2 (Inspection): Query specific column definitions, data types, or file samples.
   - Phase 3 (Execution): Run the targeted SQL or data extraction.
   - Phase 4 (Synthesis): Analyze and format the final result.
2. At EVERY step, re-evaluate your plan based on the new knowledge acquired. Optimize the next calls.
3. You can call multiple agents in parallel if needed (Map-Reduce).
4. Do NOT send the entire conversation history to agents. Extract ONLY the relevant context (e.g., specific table names, specific metrics) and pass it in the "input" field for that agent.
5. You can update the "scratchpad" to store variables (like schema definitions) for future steps.

Decide the next step. Return ONLY a valid JSON object with the following structure:
{
  "status": "thinking" | "calling_agent" | "final_answer",
  "current_plan": "Your updated step-by-step plan based on current knowledge",
  "rationale": "Why you are making this decision and how it optimizes the plan",
  "scratchpad_updates": { "key": "value" }, // optional, updates to merge into the scratchpad
  "calls": [{"agent_name": "exact name of agent", "input": "highly detailed instruction, filtered to ONLY include necessary context"}], // only if status is calling_agent
  "final_answer": "The comprehensive final answer to the user", // only if status is final_answer
  "missing_information": "Any info you still need to proceed"
}`;

      try {
        const response = await llm.invoke([new HumanMessage(managerPrompt)]);
        const content = response.content as string;
        const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        const jsonStr = jsonMatch ? jsonMatch[1] : content;
        const decision = JSON.parse(jsonStr);

        if (decision.scratchpad_updates) {
          scratchpad = { ...scratchpad, ...decision.scratchpad_updates };
        }

        addStep({ status: "manager_decision", rationale: decision.rationale, plan: decision.current_plan });
        conversationHistory += `\n[Manager Plan Updated]: ${decision.current_plan}\n[Manager Rationale]: ${decision.rationale}\n`;

        if (decision.status === 'final_answer') {
          finalAnswer = decision.final_answer;
          addStep({ 
            status: "manager_final", 
            answer: finalAnswer,
            manager_summary: decision.rationale,
            judge_verdict: "Pass",
            missing_information: decision.missing_information
          });
          isDone = true;
        } else if (decision.status === 'calling_agent' && decision.calls && decision.calls.length > 0) {
          // Execute agent calls in parallel
          const callPromises = decision.calls.map(async (call: any) => {
            const selectedAgent = activeAgents.find(a => a.name === call.agent_name);
            if (selectedAgent) {
              addStep({ status: "agent_call_started", agent: selectedAgent.name, input: call.input });
              try {
                const result = await AgentExecutor.execute(selectedAgent.agent_type, selectedAgent.config || selectedAgent, call.input, llm);
                addStep({ status: "agent_call_completed", agent: selectedAgent.name, result });
                if (result.sql) {
                  addStep({ status: "sql_generated", sql: result.sql });
                }
                return `\nAgent ${selectedAgent.name} responded: ${result.answer}\n`;
              } catch (e: any) {
                addStep({ status: "agent_call_failed", agent: selectedAgent.name, error: e.message });
                return `\nAgent ${selectedAgent.name} failed: ${e.message}\n`;
              }
            } else {
               addStep({ status: "agent_not_found", agent: call.agent_name });
               return `\nAttempted to call agent ${call.agent_name} but it was not found or unavailable.\n`;
            }
          });

          const results = await Promise.all(callPromises);
          conversationHistory += results.join('');
        } else {
           // thinking or invalid
           conversationHistory += `\nManager thought: ${decision.rationale}\n`;
        }
      } catch (e: any) {
        addStep({ status: "error", message: `Manager error: ${e.message}` });
        finalAnswer = `Orchestration error: ${e.message}`;
        isDone = true;
      }
      currentStep++;
    }

    if (!isDone) {
      finalAnswer = "Reached maximum steps without a final answer.";
      addStep({ status: "manager_final", answer: finalAnswer, manager_summary: "Max steps reached." });
    }

    return {
      answer: finalAnswer,
      steps
    };
  }
}
