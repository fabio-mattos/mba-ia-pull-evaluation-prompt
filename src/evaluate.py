"""
Script COMPLETO para avaliar prompts otimizados.

Este script:
1. Carrega dataset de avaliação de arquivo .jsonl (datasets/bug_to_user_story.jsonl)
2. Cria/atualiza dataset no LangSmith
3. Puxa prompts otimizados do LangSmith Hub (fonte única de verdade)
4. Executa prompts contra o dataset
5. Calcula 4 métricas específicas para Bug to User Story:
   - Tone Score: Tom profissional e empático
   - Acceptance Criteria Score: Qualidade dos critérios de aceitação
   - User Story Format Score: Formato correto (Como... Eu quero... Para que...)
   - Completeness Score: Completude e contexto técnico
6. Publica resultados no dashboard do LangSmith
7. Exibe resumo no terminal

Suporta múltiplos providers de LLM:
- OpenAI (gpt-5, gpt-5-mini) - Constitution requirement
- Google Gemini (gemini-2.5-flash) - Alternative

Configure o provider no arquivo .env através da variável LLM_PROVIDER.
"""

import os
import sys
import json
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import (
    evaluate_tone_score,
    evaluate_acceptance_criteria_score,
    evaluate_user_story_format_score,
    evaluate_completeness_score
)

load_dotenv()


def get_llm():
    return get_configured_llm(temperature=1.0)


def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []

    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Ignorar linhas vazias
                    example = json.loads(line)
                    examples.append(example)

        return examples

    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        print("\nCertifique-se de que o arquivo datasets/bug_to_user_story.jsonl existe.")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        return dataset_name

    try:
        datasets = client.list_datasets(dataset_name=dataset_name)
        existing_dataset = None

        for ds in datasets:
            if ds.name == dataset_name:
                existing_dataset = ds
                break

        if existing_dataset:
            return dataset_name
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)

            for example in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=example["inputs"],
                    outputs=example["outputs"]
                )

            return dataset_name

    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")
        return dataset_name


def pull_prompt_from_langsmith(prompt_name: str) -> ChatPromptTemplate:
    try:
        prompt = hub.pull(prompt_name)
        return prompt

    except Exception as e:
        error_msg = str(e).lower()

        print(f"\n{'=' * 70}")
        print(f"❌ ERRO: Não foi possível carregar o prompt '{prompt_name}'")
        print(f"{'=' * 70}\n")

        if "not found" in error_msg or "404" in error_msg:
            print("⚠️  O prompt não foi encontrado no LangSmith Hub.\n")
            print("AÇÕES NECESSÁRIAS:")
            print("1. Verifique se você já fez push do prompt otimizado:")
            print(f"   python src/push_prompts.py")
            print()
            print("2. Confirme se o prompt foi publicado com sucesso em:")
            print(f"   https://smith.langchain.com/prompts")
            print()
            print(f"3. Certifique-se de que o nome do prompt está correto: '{prompt_name}'")
            print()
            print("4. Se você alterou o prompt no YAML, refaça o push:")
            print(f"   python src/push_prompts.py")
        else:
            print(f"Erro técnico: {e}\n")
            print("Verifique:")
            print("- LANGSMITH_API_KEY está configurada corretamente no .env")
            print("- Você tem acesso ao workspace do LangSmith")
            print("- Sua conexão com a internet está funcionando")

        print(f"\n{'=' * 70}\n")
        raise


def evaluate_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any
) -> Dict[str, Any]:
    try:
        inputs = example.inputs if hasattr(example, 'inputs') else {}
        outputs = example.outputs if hasattr(example, 'outputs') else {}

        chain = prompt_template | llm

        response = chain.invoke(inputs)
        answer = response.content

        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        if isinstance(inputs, dict):
            question = inputs.get("question", inputs.get("bug_report", inputs.get("pr_title", "N/A")))
        else:
            question = "N/A"

        return {
            "answer": answer,
            "reference": reference,
            "question": question
        }

    except Exception as e:
        print(f"      ⚠️  Erro ao avaliar exemplo: {e}")
        import traceback
        print(f"      Traceback: {traceback.format_exc()}")
        return {
            "answer": "",
            "reference": "",
            "question": ""
        }


def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client,
    max_examples: int = 10
) -> Dict[str, float]:
    try:
        prompt_template = pull_prompt_from_langsmith(prompt_name)

        examples = list(client.list_examples(dataset_name=dataset_name))

        llm = get_llm()

        tone_scores = []
        acceptance_criteria_scores = []
        format_scores = []
        completeness_scores = []

        for i, example in enumerate(examples[:max_examples], 1):
            result = evaluate_prompt_on_example(prompt_template, example, llm)

            if result["answer"]:
                bug_report = result["question"]
                user_story = result["answer"]
                reference = result["reference"]

                tone = evaluate_tone_score(bug_report, user_story, reference)
                acceptance = evaluate_acceptance_criteria_score(bug_report, user_story, reference)
                format_score = evaluate_user_story_format_score(bug_report, user_story, reference)
                completeness = evaluate_completeness_score(bug_report, user_story, reference)

                tone_scores.append(tone["score"])
                acceptance_criteria_scores.append(acceptance["score"])
                format_scores.append(format_score["score"])
                completeness_scores.append(completeness["score"])

        avg_tone = sum(tone_scores) / len(tone_scores) if tone_scores else 0.0
        avg_acceptance = sum(acceptance_criteria_scores) / len(acceptance_criteria_scores) if acceptance_criteria_scores else 0.0
        avg_format = sum(format_scores) / len(format_scores) if format_scores else 0.0
        avg_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

        return {
            "tone": round(avg_tone, 4),
            "acceptance_criteria": round(avg_acceptance, 4),
            "format": round(avg_format, 4),
            "completeness": round(avg_completeness, 4)
        }

    except Exception as e:
        print(f"   ❌ Erro na avaliação: {e}")
        return {
            "tone": 0.0,
            "acceptance_criteria": 0.0,
            "format": 0.0,
            "completeness": 0.0
        }


def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("=" * 32)
    print(f"Prompt: {prompt_name}")
    print(f"- Tone Score: {scores['tone']:.2f}")
    print(f"- Acceptance Criteria: {scores['acceptance_criteria']:.2f}")
    print(f"- Format Score: {scores['format']:.2f}")
    print(f"- Completeness: {scores['completeness']:.2f}")
    print("=" * 32)

    # Verifica se TODAS as métricas individuais passam
    all_passed = all(score >= 0.9 for score in scores.values())

    if all_passed:
        print("Status: APROVADO ✓ - Todas as métricas >= 0.9")
    else:
        print("Status: FALHOU - Métricas abaixo do mínimo de 0.9")
        failed_metrics = [name for name, score in scores.items() if score < 0.9]
        print(f"Métricas que falharam: {', '.join(failed_metrics)}")

    return all_passed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Avalia prompts contra dataset de bugs"
    )
    parser.add_argument(
        "--prompt",
        default="leonanluppi/bug_to_user_story_v1",
        help="Nome do prompt no Hub para avaliar (default: leonanluppi/bug_to_user_story_v1)"
    )
    
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="Número máximo de exemplos a processar (útil para testes rápidos)"
    )

    args = parser.parse_args()
    
    print("Executando avaliação dos prompts...")

    provider = os.getenv("LLM_PROVIDER", "openai")
    llm_model = os.getenv("LLM_MODEL", "gpt-5-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-5")

    required_vars = ["LANGSMITH_API_KEY", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    print("a chave openai é ", os.getenv("OPENAI_API_KEY"))
    if not check_env_vars(required_vars):
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "prompt-optimization-challenge-resolved")

    jsonl_path = "datasets/bug_to_user_story.jsonl"

    if not Path(jsonl_path).exists():
        print(f"❌ Arquivo de dataset não encontrado: {jsonl_path}")
        return 1

    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    prompts_to_evaluate = [args.prompt]

    all_passed = True
    evaluated_count = 0
    results_summary = []

    for prompt_name in prompts_to_evaluate:
        evaluated_count += 1

        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client, max_examples=args.max_examples)

            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed

            results_summary.append({
                "prompt": prompt_name,
                "scores": scores,
                "passed": passed
            })

        except Exception as e:
            all_passed = False

            results_summary.append({
                "prompt": prompt_name,
                "scores": {
                    "tone": 0.0,
                    "acceptance_criteria": 0.0,
                    "format": 0.0,
                    "completeness": 0.0
                },
                "passed": False
            })

    if all_passed:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
