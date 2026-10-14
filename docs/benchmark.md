# Benchmark

[Per-species results (CSV)](benchmark-data/benchmark.csv) ·
[Mean results (CSV)](benchmark-data/benchmark-summary.csv)

**Benchmark hardware:** 32 CPU threads + 2 × NVIDIA RTX 4090 D GPUs.

## Mean performance

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Mean&nbsp;(20&nbsp;species) | OrionGeno&#8209;E | **84.29** | 89.02 | **86.42** | **64.28** | 68.56 | **65.78** | **51.98** | **57.36** | **53.52** | 2:25:47 | — |
|  | OrionGeno | 78.81 | 72.38 | 75.08 | 52.32 | **69.58** | 57.45 | 37.32 | 52.78 | 41.76 | 0:22:21 | Clade&#8209;specific |
|  | EviANN | 75.53 | **89.59** | 81.20 | 57.28 | 67.12 | 60.95 | 46.84 | 55.20 | 49.55 | 1:54:42 | — |
|  | BRAKER4 | 80.23 | 75.79 | 77.31 | 57.73 | 66.78 | 60.12 | 39.20 | 52.45 | 43.33 | 18:03:23 | — |
|  | ANNEVO | 65.66 | 67.68 | 65.54 | 42.30 | 65.48 | 49.65 | 26.84 | 47.24 | 32.60 | 0:29:06 | Clade&#8209;specific |
|  | Tiberius | 65.32 | 64.49 | 64.66 | 43.02 | 62.32 | 49.59 | 29.02 | 46.02 | 34.29 | 0:39:09 | Clade&#8209;specific |

![Mean precision and sensitivity across 20 species](figures/benchmark-overview.svg)

[View the overview as SVG](figures/benchmark-overview.svg).

## Per-species performance

### Mammals

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Homo&nbsp;sapiens* | OrionGeno&#8209;E | **92.4** | **84.6** | **88.3** | **42.2** | 52.9 | **46.9** | **25.1** | **42.5** | **31.6** | 4:12:15 | — |
|  | OrionGeno | 82.1 | 72.7 | 77.1 | 18.6 | **70.9** | 29.5 | 7.2 | 39.7 | 12.2 | 1:17:38 | `Mammalia` |
|  | EviANN | 84.1 | 82.1 | 83.1 | 39.3 | 50.7 | 44.3 | 24.0 | 39.3 | 29.8 | 2:36:11 | — |
|  | BRAKER4 | 78.4 | 67.1 | 72.3 | 22.3 | 62.4 | 32.9 | 9.5 | 39.3 | 15.3 | 94:39:38 | — |
|  | ANNEVO | 73.3 | 64.3 | 68.5 | 16.5 | 62.6 | 26.1 | 7.1 | 38.8 | 12.0 | 2:32:10 | `ANNEVO_Mammalia.pt` |
|  | Tiberius | 72.4 | 62.2 | 66.9 | 16.4 | 60.8 | 25.8 | 6.9 | 37.1 | 11.6 | 1:58:45 | `mammalia_nosofttmasking_v2` |
| *Mus&nbsp;musculus* | OrionGeno&#8209;E | **91.7** | **90.9** | **91.3** | **45.5** | 68.8 | **54.8** | **30.4** | **58.1** | **39.9** | 3:39:27 | — |
|  | OrionGeno | 82.2 | 77.9 | 80.0 | 25.8 | **76.1** | 38.5 | 10.8 | 45.2 | 17.4 | 1:01:08 | `Mammalia` |
|  | EviANN | 84.9 | 90.8 | 87.8 | 42.2 | 68.3 | 52.2 | 28.8 | 56.0 | 38.0 | 2:24:15 | — |
|  | BRAKER4 | 76.5 | 76.8 | 76.6 | 31.1 | 71.1 | 43.3 | 13.4 | 44.0 | 20.5 | 35:06:27 | — |
|  | ANNEVO | 73.6 | 64.8 | 68.9 | 22.9 | 62.8 | 33.6 | 10.5 | 40.6 | 16.7 | 2:01:03 | `ANNEVO_Mammalia.pt` |
|  | Tiberius | 72.9 | 66.0 | 69.3 | 22.9 | 64.3 | 33.8 | 10.3 | 41.0 | 16.5 | 2:02:18 | `mammalia_nosofttmasking_v2` |
| *Rattus&nbsp;norvegicus* | OrionGeno&#8209;E | **86.1** | **91.5** | **88.7** | **48.9** | 67.1 | **56.6** | **32.5** | **57.3** | **41.5** | 5:34:03 | — |
|  | OrionGeno | 76.3 | 76.8 | 76.5 | 26.3 | **74.7** | 38.9 | 10.9 | 46.5 | 17.7 | 1:02:47 | `Mammalia` |
|  | EviANN | 81.1 | 90.5 | 85.5 | 46.3 | 66.7 | 54.7 | 31.2 | 55.0 | 39.8 | 4:07:11 | — |
|  | BRAKER4 | 75.1 | 79.4 | 77.2 | 34.7 | 74.3 | 47.3 | 14.2 | 46.4 | 21.7 | 69:43:57 | — |
|  | ANNEVO | 68.6 | 63.9 | 66.2 | 23.6 | 61.9 | 34.2 | 10.4 | 41.1 | 16.6 | 2:19:57 | `ANNEVO_Mammalia.pt` |
|  | Tiberius | 67.0 | 62.8 | 64.8 | 23.1 | 60.9 | 33.5 | 10.2 | 40.4 | 16.3 | 2:19:29 | `mammalia_nosofttmasking_v2` |

### Other animals

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Caenorhabditis&nbsp;elegans* | OrionGeno&#8209;E | 77.8 | 88.4 | **82.8** | **58.8** | 65.4 | 61.9 | **58.5** | 60.0 | 59.2 | 0:35:28 | — |
|  | OrionGeno | 55.5 | 52.8 | 54.1 | 36.0 | 49.1 | 41.5 | 33.3 | 44.8 | 38.2 | 0:04:23 | `Other_invertebrata` |
|  | EviANN | 67.7 | **92.2** | 78.1 | 51.2 | 68.0 | 58.4 | 51.5 | 60.9 | 55.8 | 0:27:03 | — |
|  | BRAKER4 | **79.1** | 86.7 | 82.7 | 57.5 | **72.7** | **64.2** | 57.6 | **74.9** | **65.1** | 3:37:24 | — |
|  | ANNEVO | 40.0 | 44.0 | 41.9 | 26.0 | 41.0 | 31.8 | 26.5 | 41.4 | 32.3 | 0:01:30 | `ANNEVO_Insecta.pt` |
|  | Tiberius | 45.7 | 54.2 | 49.6 | 30.2 | 51.3 | 38.0 | 30.3 | 50.9 | 38.0 | 0:03:55 | `insecta` |
| *Danio&nbsp;rerio* | OrionGeno&#8209;E | **82.6** | **91.2** | **86.7** | **44.1** | 72.2 | **54.8** | **29.7** | **63.9** | **40.6** | 3:10:02 | — |
|  | OrionGeno | 75.3 | 78.1 | 76.7 | 30.0 | **75.7** | 43.0 | 12.5 | 42.9 | 19.4 | 0:36:51 | `Actinopterygii` |
|  | EviANN | 74.6 | 90.2 | 81.7 | 39.9 | 71.3 | 51.2 | 27.9 | 62.0 | 38.5 | 2:20:02 | — |
|  | BRAKER4 | 75.8 | 82.0 | 78.8 | 37.8 | 74.9 | 50.2 | 16.2 | 45.3 | 23.9 | 49:19:42 | — |
|  | ANNEVO | 72.9 | 74.3 | 73.6 | 29.1 | 72.1 | 41.5 | 12.6 | 42.8 | 19.5 | 1:06:15 | `ANNEVO_Actinopteri.pt` |
|  | Tiberius | 66.0 | 62.5 | 64.2 | 26.3 | 60.6 | 36.7 | 11.3 | 35.6 | 17.2 | 1:03:26 | `vertebrates` |
| *Drosophila&nbsp;melanogaster* | OrionGeno&#8209;E | **90.0** | **96.4** | **93.1** | **69.3** | 74.0 | **71.6** | **60.5** | 58.7 | **59.6** | 0:45:58 | — |
|  | OrionGeno | 81.0 | 80.1 | 80.5 | 49.3 | **76.6** | 60.0 | 33.2 | **59.0** | 42.5 | 0:05:34 | `Arthopoda` |
|  | EviANN | 82.5 | 95.4 | 88.5 | 64.6 | 73.0 | 68.5 | 56.9 | 56.5 | 56.7 | 0:37:06 | — |
|  | BRAKER4 | 87.6 | 82.1 | 84.8 | 62.8 | 72.5 | 67.3 | 37.4 | 52.1 | 43.5 | 4:20:00 | — |
|  | ANNEVO | 82.0 | 66.6 | 73.5 | 50.2 | 64.2 | 56.3 | 32.7 | 48.1 | 38.9 | 0:05:46 | `ANNEVO_Insecta.pt` |
|  | Tiberius | 82.6 | 73.2 | 77.6 | 51.1 | 71.2 | 59.5 | 33.0 | 53.1 | 40.7 | 0:07:55 | `insecta` |
| *Gallus&nbsp;gallus* | OrionGeno&#8209;E | **82.9** | **88.2** | **85.5** | **48.5** | 54.8 | **51.5** | **32.2** | **44.9** | **37.5** | 2:02:49 | — |
|  | OrionGeno | 73.9 | 75.4 | 74.6 | 26.9 | **72.3** | 39.2 | 11.2 | 44.1 | 17.9 | 0:27:55 | `Aves` |
|  | EviANN | 79.1 | 85.5 | 82.2 | 46.1 | 53.2 | 49.4 | 30.8 | 41.6 | 35.4 | 1:19:20 | — |
|  | BRAKER4 | 77.1 | 72.8 | 74.9 | 36.2 | 64.3 | 46.3 | 16.0 | 42.6 | 23.3 | 22:40:58 | — |
|  | ANNEVO | 67.7 | 69.2 | 68.4 | 24.6 | 66.5 | 35.9 | 11.0 | 43.5 | 17.6 | 0:14:48 | `ANNEVO_Aves.pt` |
|  | Tiberius | 63.8 | 59.4 | 61.5 | 23.2 | 57.1 | 33.0 | 10.4 | 37.2 | 16.3 | 0:50:14 | `vertebrates` |

### Flowering plants

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Arabidopsis&nbsp;thaliana* | OrionGeno&#8209;E | **86.5** | **93.4** | **89.8** | **63.2** | 75.5 | **68.8** | **53.5** | 66.3 | **59.2** | 0:59:58 | — |
|  | OrionGeno | 85.8 | 82.3 | 84.0 | 56.8 | 81.1 | 66.8 | 38.2 | 62.5 | 47.4 | 0:04:54 | `Angiosperm` |
|  | EviANN | 82.2 | 92.5 | 87.0 | 59.8 | 73.2 | 65.8 | 50.4 | 62.3 | 55.7 | 0:46:33 | — |
|  | BRAKER4 | 82.8 | 82.3 | 82.5 | 57.4 | 74.4 | 64.8 | 38.4 | 59.2 | 46.6 | 2:28:55 | — |
|  | ANNEVO | 81.4 | 90.6 | 85.8 | 53.8 | **89.1** | 67.1 | 36.3 | **69.0** | 47.6 | 0:01:29 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 81.0 | 88.4 | 84.5 | 53.5 | 86.8 | 66.2 | 35.9 | 66.9 | 46.7 | 0:05:57 | `angiosperms` |
| *Brachypodium&nbsp;distachyon* | OrionGeno&#8209;E | **86.2** | 82.4 | **84.3** | **71.5** | 61.4 | 66.1 | **60.0** | 52.8 | **56.2** | 4:53:02 | — |
|  | OrionGeno | 84.8 | 64.3 | 73.1 | 64.6 | 62.1 | 63.3 | 40.9 | 46.0 | 43.3 | 0:09:34 | `Angiosperm` |
|  | EviANN | 74.8 | **87.4** | 80.6 | 62.1 | 60.1 | 61.1 | 53.9 | 50.1 | 51.9 | 4:32:13 | — |
|  | BRAKER4 | 83.6 | 54.6 | 66.1 | 67.0 | 49.0 | 56.6 | 39.4 | 35.8 | 37.5 | 7:01:35 | — |
|  | ANNEVO | 82.0 | 82.3 | 82.1 | 62.9 | **80.0** | **70.4** | 36.7 | **54.7** | 43.9 | 0:02:43 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 70.7 | 66.2 | 68.4 | 54.2 | 64.3 | 58.8 | 30.9 | 43.0 | 36.0 | 0:10:41 | `angiosperms` |
| *Glycine&nbsp;max* | OrionGeno&#8209;E | **88.0** | **86.8** | **87.4** | **78.4** | 68.2 | **72.9** | **64.2** | **57.5** | **60.7** | 3:30:37 | — |
|  | OrionGeno | 80.1 | 72.9 | 76.3 | 60.4 | 71.4 | 65.4 | 35.9 | 51.6 | 42.3 | 0:27:12 | `Angiosperm` |
|  | EviANN | 86.0 | 86.5 | 86.2 | 76.4 | 66.7 | 71.2 | 62.6 | 54.6 | 58.3 | 2:47:22 | — |
|  | BRAKER4 | 84.9 | 54.3 | 66.2 | 68.7 | 49.8 | 57.7 | 39.3 | 36.1 | 37.6 | 16:26:46 | — |
|  | ANNEVO | 78.6 | 74.9 | 76.7 | 59.0 | 73.1 | 65.3 | 33.6 | 50.7 | 40.4 | 0:10:07 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 77.4 | 75.7 | 76.5 | 58.2 | **73.9** | 65.1 | 32.8 | 50.7 | 39.8 | 0:36:21 | `angiosperms` |
| *Oryza&nbsp;sativa* | OrionGeno&#8209;E | **85.5** | 82.1 | **83.8** | **75.3** | 64.1 | 69.3 | **62.9** | **55.6** | **59.0** | 2:17:17 | — |
|  | OrionGeno | 81.9 | 49.8 | 61.9 | 64.2 | 48.3 | 55.1 | 40.6 | 36.1 | 38.2 | 0:12:22 | `Angiosperm` |
|  | EviANN | 80.7 | **87.1** | **83.8** | 70.8 | 65.3 | 67.9 | 59.4 | 54.1 | 56.6 | 1:54:00 | — |
|  | BRAKER4 | 82.1 | 75.9 | 78.9 | 68.4 | 67.7 | 68.0 | 39.9 | 49.2 | 44.1 | 4:47:04 | — |
|  | ANNEVO | 80.3 | 80.2 | 80.2 | 63.2 | **78.2** | **69.9** | 36.7 | 53.6 | 43.6 | 0:03:57 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 74.7 | 70.3 | 72.4 | 58.8 | 68.5 | 63.3 | 33.7 | 46.4 | 39.0 | 0:14:31 | `angiosperms` |
| *Populus&nbsp;trichocarpa* | OrionGeno&#8209;E | **91.4** | 82.1 | **86.5** | **72.7** | 52.6 | 61.0 | **61.7** | 44.7 | **51.8** | 2:44:00 | — |
|  | OrionGeno | 87.7 | 70.1 | 77.9 | 59.1 | 69.1 | 63.7 | 43.7 | **60.8** | 50.9 | 0:13:07 | `Angiosperm` |
|  | EviANN | 89.0 | 80.4 | 84.5 | 70.6 | 50.5 | 58.9 | 59.5 | 42.2 | 49.4 | 2:30:57 | — |
|  | BRAKER4 | 89.8 | 72.9 | 80.5 | 66.5 | 63.8 | 65.1 | 44.3 | 52.1 | 47.9 | 13:22:55 | — |
|  | ANNEVO | 86.2 | **83.3** | 84.7 | 58.2 | **82.1** | **68.1** | 33.8 | 56.8 | 42.4 | 0:04:14 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 84.3 | 77.2 | 80.6 | 56.7 | 76.0 | 64.9 | 32.9 | 52.4 | 40.4 | 0:16:44 | `angiosperms` |
| *Solanum&nbsp;lycopersicum* | OrionGeno&#8209;E | **77.4** | **86.2** | **81.6** | **67.7** | 68.4 | **68.0** | **56.1** | **60.7** | **58.3** | 2:22:30 | — |
|  | OrionGeno | 73.4 | 71.4 | 72.4 | 56.6 | 70.0 | 62.6 | 33.4 | 50.5 | 40.2 | 0:20:11 | `Angiosperm` |
|  | EviANN | 74.9 | 84.0 | 79.2 | 65.2 | 66.1 | 65.6 | 54.4 | 57.0 | 55.7 | 1:53:35 | — |
|  | BRAKER4 | 77.2 | 52.4 | 62.4 | 63.1 | 48.6 | 54.9 | 37.1 | 36.0 | 36.5 | 10:07:06 | — |
|  | ANNEVO | 74.0 | 79.6 | 76.7 | 56.7 | **77.5** | 65.5 | 33.4 | 55.8 | 41.8 | 0:09:33 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 70.0 | 66.2 | 68.0 | 53.4 | 64.2 | 58.3 | 31.4 | 46.1 | 37.4 | 0:29:02 | `angiosperms` |
| *Zea&nbsp;mays* | OrionGeno&#8209;E | **75.8** | 85.9 | **80.5** | **58.8** | 69.0 | **63.5** | **48.8** | **63.6** | **55.2** | 6:42:12 | — |
|  | OrionGeno | 72.2 | 52.1 | 60.5 | 50.8 | 49.9 | 50.3 | 30.2 | 36.6 | 33.1 | 0:50:35 | `Angiosperm` |
|  | EviANN | 69.9 | **89.4** | 78.5 | 53.9 | **69.2** | 60.6 | 46.2 | 62.8 | 53.2 | 5:40:35 | — |
|  | BRAKER4 | 75.0 | 56.9 | 64.7 | 56.5 | 51.7 | 54.0 | 33.0 | 38.7 | 35.6 | 12:28:16 | — |
|  | ANNEVO | 73.6 | 59.4 | 65.7 | 52.1 | 57.1 | 54.5 | 29.9 | 40.5 | 34.4 | 0:35:22 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 21.0 | 12.9 | 16.0 | 11.8 | 9.9 | 10.8 | 8.8 | 9.1 | 8.9 | 2:02:34 | `angiosperms` |

### Other plants and green algae

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Chlamydomonas&nbsp;reinhardtii* | OrionGeno&#8209;E | 66.4 | 83.3 | 73.9 | 49.8 | 59.4 | 54.2 | 55.8 | 65.4 | 60.2 | 0:26:33 | — |
|  | OrionGeno | 64.6 | 70.0 | 67.2 | 47.0 | 54.4 | 50.4 | 53.8 | 63.0 | 58.0 | 0:04:49 | `Angiosperm` |
|  | EviANN | 33.5 | **88.0** | 48.5 | 25.7 | 62.8 | 36.5 | 28.7 | 66.4 | 40.1 | 0:19:14 | — |
|  | BRAKER4 | 64.5 | 84.4 | 73.1 | 55.1 | 65.9 | 60.0 | 52.8 | **70.0** | 60.2 | 3:52:25 | — |
|  | ANNEVO | 2.0 | 13.6 | 3.5 | 1.7 | 12.2 | 3.0 | 1.1 | 7.8 | 1.9 | 0:01:29 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | **78.0** | 81.1 | **79.5** | **69.9** | **77.7** | **73.6** | **60.9** | 68.3 | **64.4** | 0:06:50 | `chlorophyta` |
| *Marchantia&nbsp;polymorpha* | OrionGeno&#8209;E | 66.3 | 87.1 | **75.3** | **61.1** | 61.1 | **61.1** | **52.2** | **43.9** | **47.7** | 1:54:06 | — |
|  | OrionGeno | 63.3 | 68.0 | 65.6 | 55.2 | **66.5** | 60.3 | 32.7 | 39.4 | 35.7 | 0:07:57 | `Angiosperm` |
|  | EviANN | 52.3 | **91.9** | 66.7 | 48.2 | 56.7 | 52.1 | 44.8 | 41.3 | 43.0 | 1:40:44 | — |
|  | BRAKER4 | **66.6** | 64.7 | 65.6 | 59.4 | 54.6 | 56.9 | 35.8 | 36.1 | 35.9 | 3:04:08 | — |
|  | ANNEVO | 41.7 | 57.1 | 48.2 | 36.2 | 55.6 | 43.9 | 20.9 | 32.1 | 25.3 | 0:02:44 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 23.0 | 31.0 | 26.4 | 20.0 | 30.3 | 24.1 | 10.3 | 15.6 | 12.4 | 0:09:14 | `angiosperms` |
| *Physcomitrium&nbsp;patens* | OrionGeno&#8209;E | **81.3** | 86.6 | **83.9** | **58.2** | 67.0 | 62.3 | **41.4** | **49.8** | **45.2** | 2:18:00 | — |
|  | OrionGeno | 76.0 | 58.8 | 66.3 | 49.1 | 58.2 | 53.3 | 18.0 | 30.4 | 22.6 | 0:15:17 | `Angiosperm` |
|  | EviANN | 70.5 | **89.8** | 79.0 | 50.3 | 64.5 | 56.5 | 38.0 | 47.7 | 42.3 | 1:55:22 | — |
|  | BRAKER4 | **81.3** | 84.8 | 83.0 | 55.8 | **70.7** | **62.4** | 25.0 | 48.6 | 33.0 | 4:31:40 | — |
|  | ANNEVO | 54.0 | 60.3 | 57.0 | 34.6 | 59.3 | 43.7 | 14.4 | 35.1 | 20.4 | 0:04:43 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 43.8 | 46.8 | 45.3 | 28.0 | 45.8 | 34.8 | 11.3 | 26.5 | 15.8 | 0:19:31 | `angiosperms` |

### Fungi

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Saccharomyces&nbsp;cerevisiae* | OrionGeno&#8209;E | 90.5 | 98.4 | **94.3** | **89.7** | 93.0 | **91.3** | 70.0 | 74.2 | 72.0 | 0:07:00 | — |
|  | OrionGeno | **91.2** | 96.1 | 93.6 | 88.6 | 93.3 | 90.9 | **91.2** | 96.1 | **93.6** | 0:01:42 | `Fungi` |
|  | EviANN | 89.1 | **98.6** | 93.6 | 87.0 | **94.3** | 90.5 | 70.2 | 75.9 | 72.9 | 0:05:23 | — |
|  | BRAKER4 | 86.5 | 97.0 | 91.4 | 81.2 | 84.5 | 82.8 | 86.5 | 95.2 | 90.6 | 1:09:40 | — |
|  | ANNEVO | 89.7 | 97.5 | 93.4 | 85.9 | 93.4 | 89.5 | 89.7 | **97.5** | 93.4 | 0:00:25 | `ANNEVO_Fungi.pt` |
|  | Tiberius | 90.0 | 96.3 | 93.0 | 87.4 | 93.4 | 90.3 | 90.0 | 96.3 | 93.0 | 0:01:38 | `fungi` |
| *Schizosaccharomyces&nbsp;pombe* | OrionGeno&#8209;E | 91.8 | **97.5** | **94.6** | 90.2 | **90.2** | **90.2** | 65.8 | 60.7 | 63.1 | 0:08:15 | — |
|  | OrionGeno | **93.1** | 87.1 | 90.0 | **90.3** | 84.6 | 87.4 | **74.2** | **69.6** | **71.8** | 0:01:17 | `Fungi` |
|  | EviANN | 79.8 | 95.0 | 86.7 | 77.2 | 84.2 | 80.5 | 59.8 | 59.0 | 59.4 | 0:06:41 | — |
|  | BRAKER4 | 90.2 | 94.8 | 92.4 | 87.6 | 84.4 | 86.0 | 58.7 | 59.8 | 59.2 | 1:20:13 | — |
|  | ANNEVO | 87.5 | 89.3 | 88.4 | 85.2 | 87.1 | 86.1 | 55.3 | 56.5 | 55.9 | 0:00:22 | `ANNEVO_Fungi.pt` |
|  | Tiberius | 89.0 | 91.6 | 90.3 | 87.0 | 89.6 | 88.3 | 56.4 | 58.1 | 57.2 | 0:01:19 | `fungi` |

### Apicomplexa

| Species | Tool | Gene&nbsp;Sn | Gene&nbsp;Pr | Gene&nbsp;F1 | CDSs&nbsp;Sn | CDSs&nbsp;Pr | CDSs&nbsp;F1 | Transcript&nbsp;Sn | Transcript&nbsp;Pr | Transcript&nbsp;F1 | Time | Checkpoints |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| *Plasmodium&nbsp;falciparum* | OrionGeno&#8209;E | 95.2 | **97.3** | **96.2** | **91.6** | 86.0 | 88.7 | 78.3 | 66.5 | 71.9 | 0:12:17 | — |
|  | OrionGeno | **95.8** | 90.9 | 93.3 | 90.8 | **87.3** | **89.0** | **94.5** | **90.9** | **92.7** | 0:01:44 | `Protozoa` |
|  | EviANN | 73.9 | 94.4 | 82.9 | 68.8 | 77.7 | 73.0 | 57.8 | 59.4 | 58.6 | 0:10:06 | — |
|  | BRAKER4 | 90.5 | 93.9 | 92.2 | 85.4 | 78.3 | 81.7 | 89.6 | 87.6 | 88.6 | 0:58:45 | — |
|  | ANNEVO | 4.1 | 38.4 | 7.4 | 3.6 | 33.8 | 6.5 | 4.1 | 38.4 | 7.4 | 0:03:14 | `ANNEVO_Magnoliopsida.pt` |
|  | Tiberius | 33.1 | 45.8 | 38.4 | 28.4 | 39.8 | 33.1 | 32.7 | 45.8 | 38.2 | 0:02:34 | `diatoms` |

<details>
<summary>Gene-level comparison across 20 species</summary>

![Gene-level precision and sensitivity across 20 species](figures/benchmark-gene-by-species.svg)

[View the per-species figure as SVG](figures/benchmark-gene-by-species.svg).

</details>

<details>
<summary>CDS chain-level comparison across 20 species</summary>

![CDS chain-level precision and sensitivity across 20 species](figures/benchmark-cds-by-species.svg)

[View the CDS chain-level figure as SVG](figures/benchmark-cds-by-species.svg).

</details>

<details>
<summary>Transcript-level comparison across 20 species</summary>

![Transcript-level precision and sensitivity across 20 species](figures/benchmark-transcript-by-species.svg)

[View the transcript-level figure as SVG](figures/benchmark-transcript-by-species.svg).

</details>
