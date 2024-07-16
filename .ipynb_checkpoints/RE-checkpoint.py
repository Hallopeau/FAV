import statistics
import numpy as np

class RGenerator:
    """
    A class to generate a report based on evaluation metrics.

    Attributes:
        metrics: List of evaluation metrics.
    """

    def __init__(self, metrics):
        """
        Initializes the RGenerator with the given metrics.

        Args:
            metrics: List of evaluation metrics.
        """
        self.metrics = metrics

    def report(self):
        """
        Generates and prints a report based on the evaluation metrics.

        Returns:
            A list containing mean and standard deviation of precision, recall, F1-score, and IoU.
        """
        precision = []
        recall = []
        f1_score = []
        iou = []
        cm = []

        for metric in self.metrics:
            cr = metric[1]
            cr = cr.encode().decode('unicode_escape')
            lines = cr.strip().split('\n')
            class_1_line = lines[3]
            class_1_values = class_1_line.split()[1:4]

            precision.append(float(class_1_values[0]))
            recall.append(float(class_1_values[1]))
            f1_score.append(float(class_1_values[2]))

            iou.append(metric[2])
            cm.append(metric[0])

        report = [
            [statistics.mean(precision), statistics.pstdev(precision)],
            [statistics.mean(recall), statistics.pstdev(recall)],
            [statistics.mean(f1_score), statistics.pstdev(f1_score)],
            [statistics.mean(iou), statistics.pstdev(iou)]
        ]

        print(f"Precision (mean, stdev): {report[0][0]*100:.0f}%, {report[0][1]*100:.0f}%")
        print(f"Recall (mean, stdev): {report[1][0]*100:.0f}%, {report[1][1]*100:.0f}%")
        print(f"F1-score (mean, stdev): {report[2][0]*100:.0f}%, {report[2][1]*100:.0f}%")
        print(f"IoU (mean, stdev): {report[3][0]*100:.0f}%, {report[3][1]*100:.0f}%")

        mean_cm = np.mean(np.array(cm), axis=0)
        print(f"\n{np.round(mean_cm, decimals=0)}")
        std_cm = np.std(np.array(cm), axis=0)
        print(f"\n{np.round(std_cm, decimals=0)}\n")

        return report
        