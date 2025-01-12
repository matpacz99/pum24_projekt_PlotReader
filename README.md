# pum24_projekt_PlotReader
Repozytorium zawierające kod istotny dla projektu PlotReader Kacpra Siąkały i Mateusza Pączka
Plik PlotReader.txt zawiera kod przekształcający skan wykresu w dane cyfrowe.
Plik axis_detector.py to program w języku python, który trenuje model Faster r-CNN, a następnie próbuje wykryć osie wykresu opisane jako vertical_line oraz horizontal_line.
Podczas używania axis_detector.py należy w odpowiednim miejscu wpisać ścieżkę do folderu z obrazami testowymi (image_dir), ścieżkę do pliku z etykietami wykonanego optymalnie w programie CVAT (annotation_file) oraz ścieżkę do obrazu z wykresem, na którym mają być wykryte osie (image_path). Należy również dostosować odpowiedni próg zaufania dla wykrywanych obiektów (treshold) oraz rekomendowaną minimalną wielkość wykrywanych obiektów (drugi argument funcji filter_small_boxes, rekomendowane 10000). Program często wykrywa jedynie jeden rodzaj osi.
