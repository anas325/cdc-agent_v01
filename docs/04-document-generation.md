# Document Generation

Two nodes handle the terminal phase: `synthesizer` assembles the document,
`final_validator` does one last QA pass and writes the report. Both run only
once, after the orchestrator determines every required section is
`complete` (or the loop limit forced a stop with non-blocking gaps
downgraded — see [Graph & Agents](02-graph-and-agents.md#loop-limits)).

## Synthesizer — [`src/agents/synthesizer.py`](../src/agents/synthesizer.py)

### Per-section rendering

`render_slot(state, section_id, items)` turns a section's accepted
`ContextItem`s into prose:

- Splits items into `assumptions` (`source == "assumption"`) and
  `normal_items` (everything else).
- One LLM call renders `normal_items` into fluid, professional French prose
  (Markdown `###` subheadings and lists allowed, no meta-commentary about the
  drafting process, no question-like phrasing — it's writing a spec, not
  summarizing a conversation).
- Assumptions are **not** blended into that prose. Each is appended
  afterward as its own Quarto callout block:

  ```
  ::: {.callout-warning}
  ## Hypothèse retenue
  ASSUMPTION: <text>
  :::
  ```

  This is deliberate and important: a reader of the final DOCX must be able
  to visually distinguish "the team told us this" from "we assumed this
  because nobody answered" without reading closely.
- If a section has no context items at all, it renders as
  `_Aucune information disponible pour cette section._` rather than calling
  the LLM.

### Template filling

`build_section_render_map(state)` iterates `sections_config`, collects every
`ContextItem` whose `section_ids` includes that section, renders its slot,
and also returns `mapped: dict[section_id, list[context_item_id]]` — this
mapping is what `final_validator` later uses to find context that fell
through the cracks.

`fill_template(template_text, slots, project_subtitle)` does plain string
substitution against [`templates/cdc_template.qmd`](../templates/cdc_template.qmd):

```
{{ project_subtitle }}   -> fixed string "Généré par CDC Refinement Agent Swarm"
{{ generation_date }}    -> today's ISO date
{{ <template_slot> }}    -> rendered prose, one per section
```

The template itself is a normal Quarto document with YAML frontmatter
(`format: docx`, `toc: true`, `number-sections: true`) and one `#` heading
per section wrapping each `{{ slot }}`. Section headings in the template are
static French titles matching `sections.yaml`'s `title` fields — adding a
new section to `sections.yaml` requires adding both a `template_slot` there
and a matching heading + placeholder in the `.qmd` template (they're not
generated from the config).

### Rendering to DOCX

`run_synthesis(state)`:

1. Writes `output/cdc_final.qmd` (always — this succeeds even without Quarto
   installed).
2. If `quarto` is found on `PATH` (`shutil.which`), shells out:
   `quarto render cdc_final.qmd --to docx`, run with `cwd=output/`.
3. Failure (`subprocess.CalledProcessError`, or Quarto absent) is swallowed
   — non-fatal. The Streamlit UI checks for the `.docx` file's existence and
   shows a "Quarto not installed / render failed, .qmd still available"
   message instead of erroring (see [Streamlit UI](05-streamlit-ui.md)).

## Final validator — [`src/agents/final_validator.py`](../src/agents/final_validator.py)

Runs immediately after synthesis, before the graph reaches `END`.

- **`find_unmapped_context(state, mapped)`** — pure set-difference: any
  *live* `ContextItem` whose id doesn't appear in any section's mapped-id list.
  This can happen if an item's `section_ids` referenced a section id that's no
  longer in `sections_config`, or more commonly, was never assigned to a
  concrete section at all. Superseded items are excluded: they were
  deliberately retired (see
  [integrate_answers](02-graph-and-agents.md#integrate_answers)) and get their
  own report section, so listing them here would read as a synthesis failure.
- **`find_final_contradictions(qmd_text)`** — one more LLM call, this time
  reading the *entire assembled document* at once, looking for
  inconsistencies invisible when each section was drafted independently
  (the example in the docstring/prompt: two sections quoting different
  numbers for the same metric, or a feature mentioned in one section and
  explicitly excluded in another). This is a deliberate second line of
  defense beyond the per-turn `critic` node, which only ever sees fresh
  context against existing context — it never re-reads the final rendered
  prose as a whole.
- **`write_qa_report(state, unmapped, final_contradictions)`** — writes
  `output/qa_report.md`, each section pulled from `state["gaps"]` by status,
  from the context items' provenance, or from the two lists above:

  ```markdown
  # Rapport QA — Cahier des charges

  ## Lacunes résolues (N)
  ## Hypothèses retenues faute de réponse (N)
  ## Lacunes reportées (limite de tours atteinte) (N)
  ## Traçabilité des informations produites par le système (N)
  ## Hypothèses écartées par une réponse ultérieure (N)
  ## Éléments de contexte non intégrés au document (N)
  ## Incohérences détectées lors de la relecture finale (N)
  ## Motif d'arrêt          <- only if state.stop_reason is set
  ```

  "Résolues" covers gap statuses `resolved`, `rag_answered`, and
  `user_answered` together — the report doesn't distinguish how a gap was
  closed, only that it was. "Hypothèses écartées" is the counterpart to
  superseding: an assumption that a later user answer overruled is gone from
  the document but named here, alongside what replaced it.

Finally, `final_validator_node` (in `graph.py`) sets `done=True`, which is
what routes the graph to `END` and flips the Streamlit UI into its
completion view.

## Example output

`output/qa_report.md` in this repo (from a prior sample run against
`data/cahier_des_charges_exemple.md`) shows the mechanism catching real
issues: two sections quoting different percentage breakdowns for the same
metric, and a status value (`"Non assigné"`) referenced in the functional
spec but missing from the technical spec's `tickets.status` enum — exactly
the class of contradiction `find_final_contradictions` is meant to surface.
