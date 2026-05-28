# Handout — Multi-Table JOIN Usability

## Problem Statement

Query Studio loads multiple CSVs but doesn't make JOINs obvious or easy. Users load 3 files thinking "I need to join these" but the UI shows 3 disconnected schemas with no visual indication of how they connect.

**Current experience:**
- Sidebar shows each table's schema separately (products, suppliers, received_inventory)
- No visual relationship diagram showing which columns are joinable
- Visual composer is single-table only (SELECT/WHERE/GROUP BY/HAVING/ORDER BY) — no JOIN section
- JOINs only work via: (a) pre-built showcase SQL, (b) NL Ask with LLM, (c) quick query buttons
- Column selection doesn't show which table each column belongs to

**User mental model:** "I have 3 Excel files, I need to join them" → but the parts are not joined in the UI.

## Scope

Fix the multi-table JOIN experience so users can:
1. See how tables relate (visual relationship diagram)
2. Build JOINs visually (pick tables, pick join columns, pick join type)
3. Select columns with table context (show which table each column belongs to)
4. Understand the JOIN clause being built (preview the SQL)

This is Phase 5 (multi-source) from the roadmap.

## Proposed Solutions

### 1. Relationship Visualization

**What:** Show detected FK relationships between tables in the sidebar or a dedicated panel.

**How:**
- Auto-detect joinable columns by name (product_id, supplier_id)
- Display a simple diagram: `products.product_id ↔ received_inventory.product_id`
- Show joinable columns with a visual indicator (link icon, colored badge)
- Click a relationship to pre-fill the JOIN composer

**Why:** Users need to see which columns connect before they can build JOINs.

### 2. Visual JOIN Composer

**What:** Add a JOIN section to the composer (or a separate JOIN composer dialog).

**How:**
- Pick left table, right table
- Pick join columns (dropdown filtered to matching types)
- Pick join type (INNER, LEFT, RIGHT, FULL)
- Add multiple JOINs (for 3+ tables)
- Preview the JOIN clause as you build it

**Why:** The composer is the visual query builder — JOINs should be part of it.

### 3. Table-Qualified Column Selection

**What:** Show which table each column belongs to when selecting columns.

**How:**
- Change column selector from `product_name` to `products.product_name`
- Group columns by table in the multiselect
- Show table name as a prefix or badge
- Handle ambiguous column names (e.g., both tables have `id`)

**Why:** Users need to know which table a column comes from, especially after JOINs.

### 4. JOIN Clause Preview

**What:** Show the JOIN clause being built in real-time.

**How:**
- Display the generated JOIN SQL below the composer
- Highlight which tables and columns are being joined
- Show the join type (INNER, LEFT, etc.)
- Update as the user changes the JOIN configuration

**Why:** Users need to see what SQL is being generated to trust it.

## Implementation Plan

### Phase 5a: Relationship Detection (Foundation)

**Goal:** Auto-detect and display joinable columns.

**Tasks:**
1. Add FK detection logic to `src/ingestion.py` or new `src/relationships.py`
   - Match columns by name (product_id, supplier_id)
   - Match columns by type (both numeric, both text)
   - Store relationships in session state: `ss.relationships = [{left: "products.product_id", right: "received_inventory.product_id", type: "FK"}]`

2. Add relationship visualization to sidebar
   - New section below schema: "Relationships"
   - Show each relationship as a card: `products.product_id ↔ received_inventory.product_id`
   - Click to pre-fill JOIN composer (Phase 5b)

3. Add joinable column indicators to schema display
   - Badge or icon next to columns that are part of a relationship
   - Tooltip showing which table it connects to

**Files to change:**
- `src/ingestion.py` — add FK detection
- `src/streamlit_app/components/sidebar.py` — add relationship visualization
- `src/streamlit_app/state.py` — add `relationships` to session state

**Done when:**
- Loading supply chain demo shows 2 relationships (product_id, supplier_id)
- Sidebar displays relationships with visual indicators
- Schema columns show joinable badges

### Phase 5b: Visual JOIN Composer

**Goal:** Let users build JOINs visually.

**Tasks:**
1. Add JOIN section to composer (or new dialog)
   - New section: "06 JOIN"
   - Pick left table, right table (dropdowns)
   - Pick join columns (filtered to matching types)
   - Pick join type (INNER, LEFT, RIGHT, FULL)
   - Add multiple JOINs for 3+ tables

2. Update QueryModel to support JOINs
   - Add `joins: List[Join]` to QueryModel
   - Add `Join` dataclass: `left_table, left_col, right_table, right_col, join_type`
   - Update `to_sql()` to emit JOIN clauses

3. Sync JOIN composer with SQL preview
   - Regenerate SQL when JOIN changes
   - Show JOIN clause in preview

**Files to change:**
- `src/query_model.py` — add Join dataclass, update QueryModel
- `src/streamlit_app/components/composer.py` — add JOIN section
- `src/streamlit_app/components/sql_preview.py` — show JOIN clause

**Done when:**
- User can build a 2-table JOIN visually
- SQL preview shows the JOIN clause
- Query executes with JOIN

### Phase 5c: Table-Qualified Columns

**Goal:** Show which table each column belongs to.

**Tasks:**
1. Update column selector to show table prefix
   - Change `product_name` to `products.product_name`
   - Group by table in multiselect
   - Handle ambiguous names (both tables have `id`)

2. Update QueryModel to use qualified names
   - Store columns as `table.column` internally
   - Emit qualified names in SQL
   - Handle SELECT * from specific table

3. Update sidebar schema display
   - Show qualified names in schema sections
   - Highlight which table is "primary" (first in FROM)

**Files to change:**
- `src/query_model.py` — support qualified column names
- `src/streamlit_app/components/composer.py` — update column selector
- `src/streamlit_app/components/sidebar.py` — show qualified names

**Done when:**
- Column selector shows `products.product_name` instead of `product_name`
- SQL uses qualified names: `SELECT products.product_name FROM ...`
- Ambiguous columns are disambiguated

### Phase 5d: JOIN Clause Preview

**Goal:** Show the JOIN clause being built in real-time.

**Tasks:**
1. Add JOIN preview below composer
   - Display generated JOIN SQL
   - Highlight tables and columns
   - Show join type

2. Add visual JOIN diagram
   - Simple diagram showing table connections
   - Highlight active JOIN
   - Click to edit

**Files to change:**
- `src/streamlit_app/components/composer.py` — add JOIN preview
- `src/streamlit_app/styles.css` — style JOIN diagram

**Done when:**
- JOIN preview updates as user changes configuration
- Visual diagram shows table connections
- User can click diagram to edit JOIN

## Testing Strategy

**Manual testing:**
1. Load supply chain demo (3 tables)
2. Verify relationships are detected and displayed
3. Build a 2-table JOIN visually (products ↔ received_inventory)
4. Build a 3-table JOIN (products ↔ received_inventory ↔ suppliers)
5. Verify SQL preview shows correct JOIN clauses
6. Execute query and verify results
7. Test with ambiguous column names (both tables have `id`)
8. Test with different join types (INNER, LEFT, RIGHT)

**Edge cases:**
- Tables with no joinable columns
- Multiple joinable columns between same tables
- Self-joins (table joined to itself)
- Circular relationships (A ↔ B ↔ C ↔ A)

## Out of Scope

- Multi-CSV upload UI (already exists via supply chain demo)
- NL-to-SQL for JOINs (already works via `nl_to_raw_sql()`)
- Quick query templates for JOINs (already exist)
- JOIN optimization or query planning
- Visual query execution plan

## Success Criteria

- User can load 3 CSVs and immediately see how they relate
- User can build a 2-table JOIN visually without writing SQL
- User can build a 3-table JOIN visually
- User can see which table each column belongs to
- User can understand the JOIN clause being generated
- No regression in single-table experience

## Notes

- This is a real usability problem, not a showcase
- The current multi-table experience is confusing because the parts are not joined
- Phase 5 is marked as "🔜 next" in the roadmap but not started
- The visual composer is the right place for JOINs — it's the visual query builder
- FK detection should be simple (name matching) — no need for complex heuristics
- Table-qualified columns are important for clarity, especially with JOINs
