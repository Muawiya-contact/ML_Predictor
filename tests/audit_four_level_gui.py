"""Display-dependent GUI regression check using the real current bundle.

Requires tkinter/display and the pinned local SapBERT cache. Translation and
class decisions are stubbed only to exercise all four rendering branches. Model
prediction parity is checked separately against the study's held-out records.
"""
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from triage_gui import TriageGUI, LEVEL_NAMES, messagebox

failed = []
messagebox.showerror = lambda title, message, **kw: failed.append(f'{title}: {message}')
app = TriageGUI()
started = time.monotonic()


def check():
    if failed:
        print(failed, flush=True)
        app.destroy()
        return
    if app.artifacts is None:
        if time.monotonic() - started > 180:
            failed.append('Bundle load timeout')
            app.destroy()
            return
        app.after(250, check)
        return
    try:
        assert app.active_manifest()['labels'] == [0, 1, 2, 3]
        assert app.active_artifacts()['model'].classes_.tolist() == [0, 1, 2, 3]
        assert len(app.nb.tabs()) == 6
        summaries = [label for label in app._model_summary_labels if label.winfo_exists()]
        assert len(summaries) == 6
        dimension = sum(b['dim'] for b in app.active_manifest()['feature_blocks'])
        assert all(f'= {dimension} inputs' in label.cget('text') for label in summaries)
        assert all('levels 0-3' in label.cget('text') for label in summaries)
        print('PASS: all six tabs identify the active model and feature dimensions', flush=True)
        matrix_tree = app._results_confusion_tree
        assert len(matrix_tree.get_children()) == 4
        assert tuple(matrix_tree['columns']) == ('0', '1', '2', '3')
        total = sum(sum(int(v) for v in matrix_tree.item(i)['values']) for i in matrix_tree.get_children())
        assert total == app.active_manifest()['dataset']['test_rows']
        print('PASS: Results displays the current four-class held-out matrix', flush=True)
        app._set_complaint('Chest pain')
        app.update()
        numbers = {name: float(var.get()) for name, var in app.fields.items()}
        categories = {name: combo.get() for name, combo in app.combos.items()}
        payload = ('Chest pain', numbers, categories, 'Chest pain', None)
        for level in range(4):
            probability = np.full(4, .025)
            probability[level] = .925
            with patch('triage_pipeline.predict_one', return_value=(level, .925, probability)):
                app._done_prediction_worker(payload)
            app.update()
            assert app.level_text.cget('text') == f'Level {level}  -  {LEVEL_NAMES[level]}'
            assert len(app._last_proba[0]) == 4
            text = ' '.join(app.proba_canvas.itemcget(i, 'text') for i in app.proba_canvas.find_all()
                            if app.proba_canvas.type(i) == 'text')
            assert all(f'L{i}' in text for i in range(4)), text
        print('PASS: all four zero-based levels and probability bars render correctly', flush=True)
        with patch.object(app, 'translate_complaint', side_effect=AssertionError('Placeholder reached translation')):
            for value in ['', 'X', 'n/a', '   ']:
                app._set_complaint(value)
                app.update()
                app._do_predict()
                assert app.level_text.cget('text') == 'Confidence: 50%'
                assert app._last_proba is None
                assert not app.proba_canvas.find_all()
                assert 'placeholder, not a model prediction' in app.stages.get('1.0', 'end')
        print('PASS: missing complaints show an explained placeholder, no level or bars', flush=True)
        app._set_complaint('Chest pain')
        app.update()
        app.fields['Age'].set(str(numbers['Age'] + 1))
        with patch('triage_pipeline.predict_one') as predict:
            app._done_prediction_worker(payload)
            predict.assert_not_called()
        assert 'Inputs changed' in app.status.get()
        print('PASS: an outstanding prediction is discarded after patient input changes', flush=True)
        rows = pd.DataFrame({'Complaint_Text': ['test'] * 4, 'Translation_English': ['test'] * 4,
                             'Predicted_Level_0to3': [0,1,2,3], 'Predicted_Triage_Level': [0,1,2,3],
                             'Predicted_Label': LEVEL_NAMES, 'Confidence': ['test'] * 4,
                             'Notes': [''] * 4})
        app._done_batch_worker((rows, '/tmp/gui-audit-fixture'))
        levels = [int(app.batch_tree.item(i)['values'][1]) for i in app.batch_tree.get_children()]
        assert levels == [0,1,2,3], levels
        print('PASS: batch results preserve levels 0, 1, 2 and 3', flush=True)
    except Exception:
        failed.append(traceback.format_exc())
        print(failed[-1], flush=True)
    finally:
        app.destroy()


app.after(250, check)
app.mainloop()
if failed:
    raise SystemExit(1)
print('Four-level GUI audit passed.', flush=True)
