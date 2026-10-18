import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import project2train as training
from project2validate import get_metrics


class MetricsTests(unittest.TestCase):
    def test_saved_counts(self):
        true = ['focused'] * 457 + ['distracted'] * 3389
        predicted = ['focused'] * 306 + ['distracted'] * 151 + ['focused'] * 36 + ['distracted'] * 3353
        _, classes, accuracy, baseline, _, macro = get_metrics(true, predicted)
        self.assertAlmostEqual(accuracy, 3659 / 3846)
        self.assertAlmostEqual(baseline, 3389 / 3846)
        self.assertAlmostEqual(macro, 0.8694142377)
        self.assertAlmostEqual(classes['focused']['recall'], 306 / 457)

    def test_missing_prediction_class(self):
        _, classes, *_ = get_metrics(['focused', 'distracted'], ['distracted'] * 2)
        self.assertEqual(classes['focused']['f1'], 0)

    def test_invalid_inputs(self):
        for true, predicted in [([], []), (['focused'], []), (['other'], ['focused'])]:
            with self.subTest(true=true, predicted=predicted), self.assertRaises(ValueError):
                get_metrics(true, predicted)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.output = root / 'source', root / 'split'
        self.rows = []
        for i in range(8):
            label = 'c0' if i % 2 else 'c1'
            image = f'image{i}.jpg'
            path = self.source / label / image
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes([i]))
            self.rows.append(dict(subject=f'p{i}', classname=label, img=image))
        settings = patch.multiple(training, TRAIN_ROOT=str(self.source), VAL_ROOT=str(self.output), LINK_MODE='copy')
        settings.start()
        self.addCleanup(settings.stop)

    def test_reproducible_driver_disjoint_split(self):
        selected = training.choose_val_subjects(self.rows)
        self.assertEqual(selected, training.choose_val_subjects(self.rows))
        summary = training.make_split(self.rows, selected)
        self.assertEqual(sum(summary['val'].values()), 5)
        self.assertEqual(sum(summary['train'].values()), 3)
        for row in self.rows:
            split = 'val' if row['subject'] in selected else 'train'
            label = training.get_binary_label(row['classname'])
            target = self.output / split / label / row['img']
            self.assertEqual(target.read_bytes(), (self.source / row['classname'] / row['img']).read_bytes())
            other = 'train' if split == 'val' else 'val'
            self.assertFalse((self.output / other / label / row['img']).exists())

    def test_existing_output_preserved(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_text('keep me')
        with self.assertRaises(FileExistsError):
            training.make_split(self.rows, {'p0'})
        self.assertEqual(marker.read_text(), 'keep me')

    def test_invalid_sources_leave_no_output(self):
        for rows, exception in [
            (self.rows + [self.rows[0]], ValueError),
            ([dict(self.rows[0], img='../outside.jpg')], ValueError),
            ([dict(self.rows[0], img='missing.jpg')], FileNotFoundError),
            ([dict(self.rows[0], classname='other')], ValueError),
        ]:
            with self.subTest(rows=rows), self.assertRaises(exception):
                training.make_split(rows, {'p0'})
            self.assertFalse(self.output.exists())

    def test_too_few_drivers(self):
        with self.assertRaises(ValueError):
            training.choose_val_subjects(self.rows[:5])


if __name__ == '__main__':
    unittest.main()
