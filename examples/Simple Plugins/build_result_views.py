import json
from pathlib import Path

import cv2

from opencv_dataflow.blocks import builtin_registry
from opencv_dataflow.core import Graph
from opencv_dataflow.simple_classifier.graphics import draw_assessment
from opencv_dataflow.simple_classifier.model import load_classifier, read_image

host = Path(__file__).resolve().parents[2]
class_root = host / 'examples/Simple Plugins/image_classification/vegqual'
project = json.loads((class_root / 'project.json').read_text())
block = next(b for b in project['blocks'] if b['id'] == 'classifier')
model = load_classifier(class_root, block['parameters']['model'])
truth = json.loads((class_root / 'ground_truth.json').read_text())
previews = class_root / 'result_views'
previews.mkdir(exist_ok=True)
selected = {}
for row in truth:
    if row['source_split'] != 'test':
        continue
    image = read_image(class_root / row['image'])
    prediction = model.predict(image)
    status = 'correct' if prediction['label'] == row['label'] else 'incorrect'
    if status not in selected:
        cv2.imwrite(str(previews / (status + '.png')), draw_assessment(image, prediction))
        selected[status] = {'image': row['image'], 'expected': row['label'],
                            'prediction': prediction}
(previews / 'predictions.json').write_text(json.dumps(selected, indent=2) + '\n', encoding='utf-8')
seg_root = host / 'examples/Simple Plugins/pixel_segmentation/armbench_tote'
graph = Graph.load(seg_root / 'project.json', builtin_registry())
try:
    block = next(b for b in graph.blocks.values() if b.type_id == 'inspection.ocv_segment_image')
    block.parameters.update(render_overlay=True, render_heatmap=True)
    print('seg parameters', block.parameters)
    target = seg_root / 'result_views'
    target.mkdir(exist_ok=True)
    for kind in ('loaded', 'empty'):
        graph.run()
        assert block.state == 'pass', block.error
        outputs = block.last_result.outputs
        cv2.imwrite(str(target / (kind + '_overlay.png')), outputs['overlay'])
        cv2.imwrite(str(target / (kind + '_heatmap.png')), outputs['heatmap'])
        print(kind, block.last_result.message)
finally:
    graph.close()
print(json.dumps(selected, indent=2))
