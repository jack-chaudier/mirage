# Projects Mirror

This directory contains curated mirrors of related research projects used by Mirage:

- `rhun/`
- `lorien/`

These mirrors are imported with hygiene filters (no secrets, local virtualenvs, cache folders, or heavy runtime outputs).

## Refresh
Use:

```bash
scripts/sync_external_projects.sh
```

Default source paths:
- `/Users/jackg/rhun`
- `/Users/jackg/lorien`

Override with env vars if needed:

```bash
RHUN_SRC=/path/to/rhun LORIEN_SRC=/path/to/lorien scripts/sync_external_projects.sh
```
