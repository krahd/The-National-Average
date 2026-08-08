# Cross-Repository Administration

Global repository registry, cross-domain status, and the master calendar are maintained in `krahd/tom-work-admin`.

This repository remains canonical for **The National Average** as an artwork/research project: implementation, data, experiments, provenance, audiovisual source, and project-specific technical/artistic state.

Related manuscripts and publication artefacts remain canonical in `krahd/academic-writing`. Submission-specific packages belong in `krahd/professional-opportunities`; grant/funding packages belong in `krahd/grant-applications`.

## Mandatory synchronisation rule

`krahd/tom-work-admin` **must be kept current** whenever work here materially changes the project's administratively meaningful state. Updating the administration repository is part of completing the change, not optional later cleanup.

Update this repository first for substantive project changes, then update `krahd/tom-work-admin` in the same work session when any of the following changes:

- project lifecycle state, working title, scope, or major artistic/research direction;
- release/version, deployment, public/private visibility, implementation milestone, or major validation state;
- relationship to a manuscript, submission, grant, collaborator, repository, dataset, compute resource, or other cross-domain dependency;
- submission/publication/award outcome where it materially affects global project status or next actions;
- deadline, event, exhibition, presentation, freeze date, or other material cross-domain date;
- current next action or major research/production gate.

A publication or submission outcome does not redefine the project's underlying status unless the project itself changes.

## Ownership boundary

Keep substantive project state, implementation, data, provenance, audiovisual source, and project-specific evidence here. `tom-work-admin` stores only the concise cross-repository view and must point back to canonical project sources rather than duplicate them.

## Completion check

Before considering a material project-state change complete, verify that:

1. this repository reflects the substantive change;
2. `krahd/tom-work-admin` reflects any resulting global status, date, relationship, or next-action change;
3. related domain repositories are updated when the change affects manuscripts, submissions, or grants;
4. no stale cross-domain status or date remains in `tom-work-admin`.
