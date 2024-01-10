import os
import sys
import unittest

sys.path.append("..")
import nPYc

"""
Tests for checking specific data values remain the same after report functionality changes
"""


class TestSampleSummaryRegression(unittest.TestCase):

    def setUp(self):
        # Load test data specific for this purpose: we know the correct numbers.
        # This data is stored in the npc-standard-project GitHub repo


        self.data = nPYc.MSDataset(os.path.join("..", "..",
                                                "npc-standard-project",
                                                "Regression_Testing_Data",
                                                "DEVSET U RPOS xcms_regressionTesting.csv"),
                                   fileType="XCMS",
                                   sop="GenericMS",
                                   noFeatureParams=9)

        self.data.addSampleInfo(descriptionFormat="Basic CSV",
                                filePath=os.path.join("..", "..",
                                                      "npc-standard-project",
                                                      "Regression_Testing_Data",
                                                      "DEVSET U RPOS Basic CSV_regressionTesting.csv"))


    def test_setup(self):
        self.assertIsNotNone(self.data)

    def test_XCMS_metadata_report_correct(self):

        sample_summary = nPYc.reports._generateSampleReport(self.data, returnOutput=True)
        """
        Check returns against expected. sample_summary is a dictionary of dataframes with keys:
        for key in sample_summary.keys():
            print(key)
            print(sample_summary[key])
        """
        # Acquired - Totals
        self.assertEqual(sample_summary["Acquired"].loc["All", "Total"], 214)
        self.assertEqual(sample_summary["Acquired"].loc["Study Sample", "Total"], 78)
        self.assertEqual(sample_summary["Acquired"].loc["Study Reference", "Total"], 23)
        self.assertEqual(sample_summary["Acquired"].loc["Long-Term Reference", "Total"], 8)
        self.assertEqual(sample_summary["Acquired"].loc["Serial Dilution", "Total"], 92)
        self.assertEqual(sample_summary["Acquired"].loc["Blank", "Total"], 12)
        self.assertEqual(sample_summary["Acquired"].loc["Unknown", "Total"], 1)

        # Acquired - Marked for exclusion
        self.assertEqual(sample_summary["Acquired"].loc["All", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Study Sample", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Study Reference", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Long-Term Reference", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Serial Dilution", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Blank", "Marked for Exclusion"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Unknown", "Marked for Exclusion"], 0)

        # Acquired - Missing/Excluded
        self.assertEqual(sample_summary["Acquired"].loc["All", "Missing/Excluded"], 1)
        self.assertEqual(sample_summary["Acquired"].loc["Study Sample", "Missing/Excluded"], 1)
        self.assertEqual(sample_summary["Acquired"].loc["Study Reference", "Missing/Excluded"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Long-Term Reference", "Missing/Excluded"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Serial Dilution", "Missing/Excluded"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Blank", "Missing/Excluded"], 0)
        self.assertEqual(sample_summary["Acquired"].loc["Unknown", "Missing/Excluded"], 0)

        self.assertEqual(sample_summary["NoMetadata Details"].loc[0, "Sample File Name"], "PipelineTesting_RPOS_ToF10_U1W98")
        self.assertEqual(sample_summary["UnknownType Details"].loc[0, "Sample File Name"], "PipelineTesting_RPOS_ToF10_U1W98")
        self.assertEqual(sample_summary["NotAcquired"].loc[0, "Sample File Name"], "PipelineTesting_RPOS_ToF10_U1W97")
        self.assertEqual(sample_summary["Excluded Details"].loc[0, "Sample File Name"], "PipelineTesting_RPOS_ToF10_U1W97")
        self.assertEqual(sample_summary["StudySamples Exclusion Details"].loc[0, "Sample File Name"], "PipelineTesting_RPOS_ToF10_U1W97")

if __name__ == "__main__":
    unittest.main()
