#!/bin/sh

pyreverse -o dot -p LAFTR ./**/*.py

sed -E -i '' -e 's/color="green"/color="#00008b"/g' classes_LAFTR.dot
sed -E -i '' -e 's/color="green"/color="#00008b"/g' packages_LAFTR.dot

dot -Tpng classes_LAFTR.dot -o classes_LAFTR.png
dot -Tpng packages_LAFTR.dot -o packages_LAFTR.png

rm classes_LAFTR.dot packages_LAFTR.dot
