# Data notes

- The datasets were downloaded with PyTDC 0.4.1.
- The raw hERG TDC dataset contains 655 rows; the project proposal reports 648.
- The TDC raw files contain only `Drug_ID`, `Drug`, and `Y`; they do not include a source column.
- hERG_Karim curation merged 2,419 duplicate rows and removed 202 label-conflict rows.
- Duplicate parent structures were identified by the first InChIKey block, ignoring stereochemistry. A stereo-aware sensitivity check is planned.
