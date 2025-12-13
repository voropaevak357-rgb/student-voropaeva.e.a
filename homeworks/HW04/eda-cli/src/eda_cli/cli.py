from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
    plot_categorical_bar_chart,  # ← импортирована новая функция
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(6, help="Максимум числовых колонок для гистограмм."),
    top_k_categories: int = typer.Option(5, help="Количество top-значений для категориальных признаков."),
    title: str = typer.Option("EDA-отчёт", help="Заголовок отчёта в report.md."),
    min_missing_share: float = typer.Option(0.1, help="Порог доли пропусков для выделения проблемных колонок (0.0–1.0)."),
    categorical_bar_column: str = typer.Option("country", "--categorical-bar-column", help="Колонка для bar chart."),
    categorical_bar_top_n: int = typer.Option(10, "--categorical-bar-top-n", help="Число top-категорий для bar chart."),
) -> None:
    """
    Сгенерировать полный EDA-отчёт с расширенными настройками.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Обзор
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество в целом
    quality_flags = compute_quality_flags(summary, missing_df)

    # 3. Проблемные колонки по пропускам
    problematic_missing = []
    if not missing_df.empty:
        problematic_missing = missing_df[missing_df["missing_share"] >= min_missing_share].index.tolist()

    # 4. Сохраняем табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(f"- Порог пропусков для проблемных колонок: **{min_missing_share:.1%}**\n")
        if problematic_missing:
            f.write(f"- Проблемные колонки по пропускам: `{', '.join(problematic_missing)}`\n")
        else:
            f.write("- Проблемных колонок по пропускам не найдено.\n")
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n\n")

        f.write("## Гистограммы\n\n")
        f.write(f"Показаны гистограммы для первых **{max_hist_columns}** числовых колонок.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные/строковые признаки не найдены.\n\n")
        else:
            f.write("См. файлы в папке `top_categories/`.\n")
            if categorical_bar_column and categorical_bar_column in df.columns:
                f.write(f"\nДополнительно: bar chart для колонки **{categorical_bar_column}** "
                        f"(top {categorical_bar_top_n} категорий) — см. файл "
                        f"[`categorical_bar.png`](categorical_bar.png).\n")
        f.write("\n")

        f.write("## Другие артефакты\n\n")
        f.write("- Сводка по колонкам: `summary.csv`\n")
        f.write("- Пропуски: `missing.csv`, `missing_matrix.png`\n")
        f.write("- Корреляция: `correlation.csv`, `correlation_heatmap.png`\n")
        f.write("- Топ категории: папка `top_categories/`\n\n")

    # 6. Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")
    # Новый график
    if categorical_bar_column:
        plot_categorical_bar_chart(
            df,
            column=categorical_bar_column,
            out_path=out_root / "categorical_bar.png",
            top_n=categorical_bar_top_n,
        )

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Основной markdown: {md_path}")
    typer.echo("- Табличные файлы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png, categorical_bar.png")


if __name__ == "__main__":
    app()