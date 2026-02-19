"""
Testes automatizados para validação de prompts.
"""
import pytest
import yaml
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def load_prompts(file_path: str):
    """Carrega prompts do arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class TestPrompts:
    @pytest.fixture
    def prompt_v2(self):
        """Carrega o prompt otimizado v2 para os testes."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "bug_to_user_story_v2.yml"
        return load_prompts(str(prompt_path))
    
    def test_prompt_has_system_prompt(self, prompt_v2):
        """Verifica se o campo 'system_prompt' existe e não está vazio."""
        # Verificar se existe pelo menos uma mensagem do tipo system
        messages = prompt_v2.get("messages", [])
        assert messages, "Prompt não contém mensagens"
        
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        assert system_messages, "Prompt não contém system prompt"
        
        # Verificar se o conteúdo não está vazio
        system_content = system_messages[0].get("content", "").strip()
        assert system_content, "System prompt está vazio"

    def test_prompt_has_role_definition(self, prompt_v2):
        """Verifica se o prompt define uma persona (ex: "Você é um Product Manager")."""
        messages = prompt_v2.get("messages", [])
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        
        assert system_messages, "Prompt não contém system prompt"
        
        system_content = system_messages[0].get("content", "").lower()
        
        # Verificar se contém alguma definição de papel/persona
        role_keywords = ["você é", "voce e", "atue como", "você será", "voce sera"]
        has_role = any(keyword in system_content for keyword in role_keywords)
        
        assert has_role, "Prompt não define uma persona/papel claro (ex: 'Você é um...')"

    def test_prompt_mentions_format(self, prompt_v2):
        """Verifica se o prompt exige formato Markdown ou User Story padrão."""
        messages = prompt_v2.get("messages", [])
        
        # Concatenar todo o conteúdo das mensagens
        full_content = " ".join([
            msg.get("content", "").lower() 
            for msg in messages
        ])
        
        # Verificar se menciona formato
        format_keywords = [
            "markdown", "formato", "estrutura", "user story", 
            "como", "eu quero", "para que"
        ]
        
        has_format = any(keyword in full_content for keyword in format_keywords)
        assert has_format, "Prompt não menciona formato esperado (Markdown, User Story, etc.)"

    def test_prompt_has_few_shot_examples(self, prompt_v2):
        """Verifica se o prompt contém exemplos de entrada/saída (técnica Few-shot)."""
        messages = prompt_v2.get("messages", [])
        
        # Verificar se existem mensagens de exemplo (human/ai alternadas)
        human_messages = [msg for msg in messages if msg.get("role") == "human"]
        ai_messages = [msg for msg in messages if msg.get("role") == "ai"]
        
        # Few-shot requer pelo menos 1 exemplo (human + ai)
        assert len(human_messages) >= 1, "Prompt não contém mensagens de exemplo (human)"
        assert len(ai_messages) >= 1, "Prompt não contém respostas de exemplo (ai)"

    def test_prompt_no_todos(self, prompt_v2):
        """Garante que você não esqueceu nenhum `[TODO]` no texto."""
        messages = prompt_v2.get("messages", [])
        
        # Concatenar todo o conteúdo
        full_content = " ".join([
            msg.get("content", "") 
            for msg in messages
        ])
        
        # Verificar variações de TODO
        todo_patterns = ["[TODO]"]
        found_todos = [pattern for pattern in todo_patterns if pattern in full_content]
        
        assert not found_todos, f"Prompt contém TODOs pendentes: {found_todos}"

    def test_minimum_techniques(self, prompt_v2):
        """Verifica (através dos metadados do yaml) se pelo menos 2 técnicas foram listadas."""
        techniques = prompt_v2.get("techniques_applied", [])
        
        assert isinstance(techniques, list), "Campo 'techniques_applied' deve ser uma lista"
        assert len(techniques) >= 2, f"Mínimo de 2 técnicas requeridas, encontradas: {len(techniques)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])