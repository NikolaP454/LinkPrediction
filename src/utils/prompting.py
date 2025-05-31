def generate_base_prompt(
    title_a: str, title_b: str, abstact_a: str = "", abstact_b: str = ""
) -> str:
    """
    Generate a base prompt for the AI model.

    Args:
        title_a (str): The title of the first entity.
        title_b (str): The title of the second entity.
        abstact_a (str, optional): The abstract of the first entity. Defaults to an empty string.
        abstact_b (str, optional): The abstract of the second entity. Defaults to an empty string.

    Returns:
        str: The formatted base prompt.
    """

    return f"""Should "{title_a}" (Paper A) cite "{title_b}" (Paper B)?"""


def add_paper_relations(
    is_source: bool,
    cites: list[str],
    is_cited_by: list[str],
    max_papers: int = 2,
    source_prompt: str = "",
) -> str:
    """
    Add paper relations to the prompt.

    Args:
        is_source (bool): Whether the paper is a source or a target.
        cites (list[str]): List of papers cited by the source paper.
        is_cited_by (list[str]): List of papers that cite the source paper.
        max_papers (int, optional): Maximum number of papers to include in the relations. Defaults to 2.
        source_prompt (str, optional): The source prompt to append relations to. Defaults to an empty string.

    Returns:
        str: The formatted relations string.
    """

    if len(cites) == 0 and len(is_cited_by) == 0:
        return source_prompt

    if max_papers < 1:
        return source_prompt

    source_prompt += "\n\n"

    if len(cites) > 0:
        source_prompt += (
            f'{"Paper A" if is_source else "Paper B"} cites the following papers:\n'
        )

        for cite in cites[:max_papers]:
            source_prompt += f"- {cite}\n"

    if len(is_cited_by) > 0:
        source_prompt += "\n" if len(cites) > 0 else ""
        source_prompt += f'{"Paper A" if is_source else "Paper B"} is cited by the following papers:\n'

        for cited_by in is_cited_by[:max_papers]:
            source_prompt += f"- {cited_by}\n"

    return source_prompt.strip()
