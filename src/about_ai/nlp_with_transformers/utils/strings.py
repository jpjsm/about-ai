def generate_centered_title_with_marks(
    title: str,
    line_length: int = 144,
    marker: str = "#",
    clean_space: int = 2,
    skip_before: int = 2,
    skip_after: int = 1,
) -> str:
    if title is None:
        return ""

    title = title.strip()
    if len(title) == 0:
        return ""

    result: str = ""
    for _ in range(skip_before):
        result += "\n"

    title_len: int = len(title)
    is_title_len_odd = bool(title_len & 1)
    if is_title_len_odd:
        title = title + " "
        title_len += 1

    marker_run_length: int = 2
    if title_len >= line_length - 8:
        result += f"{marker*marker_run_length}{" "*clean_space}{title}{" "*clean_space}{marker*marker_run_length}"
    else:
        marker_run_length = (
            line_length - title_len - 2 * clean_space
        ) >> 1  # Fast divide by 2
        result += f"{marker*marker_run_length}{" "*clean_space}{title}{" "*clean_space}{marker*marker_run_length}"

    for _ in range(skip_after):
        result += "\n"

    return result
