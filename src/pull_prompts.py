"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub
3. Salva localmente em prompts/bug_to_user_story_v1.yml

SIMPLIFICADO: Usa serialização nativa do LangChain para extrair prompts.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()


def pull_prompts_from_langsmith(prompt_name: str = "leonanluppi/bug_to_user_story_v1", 
                                output_dir: str = "prompts") -> bool:
    """
    Faz pull do prompt do LangSmith Hub.
    
    Args:
        prompt_name: Nome do prompt no Hub (formato: owner/prompt)
        output_dir: Diretório para salvar o prompt localmente
        
    Returns:
        True se sucesso, False caso contrário
    """
    try:
        print_section_header("Pull de Prompts do LangSmith Hub")
        
        # Verificar credenciais
        if not check_env_vars(["LANGSMITH_API_KEY"]):
            return False
            
        # Fazer pull do prompt
        print(f"📥 Fazendo pull do prompt: {prompt_name}")
        prompt = hub.pull(prompt_name)
        
        # Extrair dados do prompt
        prompt_dict = {
            "name": prompt_name.split("/")[-1],
            "owner": prompt_name.split("/")[0] if "/" in prompt_name else "unknown",
            "messages": []
        }
        
        # Serializar mensagens do prompt
        if hasattr(prompt, "messages"):
            for msg in prompt.messages:
                msg_dict = {
                    "role": msg.__class__.__name__.replace("MessagePromptTemplate", "").replace("Message", "").lower(),
                    "content": msg.prompt.template if hasattr(msg.prompt, "template") else str(msg)
                }
                prompt_dict["messages"].append(msg_dict)
        
        # Criar path de saída
        output_path = Path(output_dir) / f"{prompt_dict['name']}.yml"
        
        # Salvar como YAML
        if save_yaml(prompt_dict, str(output_path)):
            print(f"✅ Prompt salvo com sucesso em: {output_path}")
            print(f"   - Nome: {prompt_dict['name']}")
            print(f"   - Owner: {prompt_dict['owner']}")
            print(f"   - Mensagens: {len(prompt_dict['messages'])}")
            return True
        else:
            print(f"❌ Erro ao salvar prompt em: {output_path}")
            return False
            
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("   Verifique se langchain está instalado: pip install langchain")
        return False
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            print(f"❌ Erro de autenticação: Verifique LANGSMITH_API_KEY no .env")
        elif "not found" in error_msg.lower() or "404" in error_msg:
            print(f"❌ Prompt não encontrado: {prompt_name}")
            print("   Verifique se o nome está correto (formato: owner/prompt)")
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            print(f"❌ Erro de rede: {error_msg}")
            print("   Verifique sua conexão com a internet")
        else:
            print(f"❌ Erro ao fazer pull do prompt: {error_msg}")
        return False


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Faz pull de prompts do LangSmith Hub"
    )
    parser.add_argument(
        "--prompt",
        default="leonanluppi/bug_to_user_story_v1",
        help="Nome do prompt no Hub (formato: owner/prompt)"
    )
    parser.add_argument(
        "--output-dir",
        default="prompts",
        help="Diretório para salvar o prompt localmente"
    )
    
    args = parser.parse_args()
    
    success = pull_prompts_from_langsmith(args.prompt, args.output_dir)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
