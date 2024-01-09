import os
import sys
import unittest

sys.path.append("..")
import nPYc

"""
Tests for checking specific data values remain the same after report functionality changes
"""
class test_sample_summary_regression(unittest.TestCase):

    def setUp(self):
        # load test data specific for this purpose: we know the correct numbers
        self.data = nPYc.MSDataset(os.path.join('..', '..',
                                                'npc-standard-project',
                                                'Regression_Testing_Data',
                                                'DEVSET U RPOS xcms_regressionTesting.csv'),
                                fileType='XCMS',
                                sop='GenericMS',
                                noFeatureParams=9)

        self.data.addSampleInfo(descriptionFormat='Basic CSV',
                                filePath=os.path.join('..', '..',
                                                      'npc-standard-project',
                                                      'Regression_Testing_Data',
                                                      'DEVSET U RPOS Basic CSV_regressionTesting.csv'))

    def test_report_samplesummary(self):

        sampleSummary = nPYc.reports._generateSampleReport(self.data, returnOutput=True)

        # Check returns against expected

        # Acquired - Totals
        assert sampleSummary['Acquired'].loc['All', 'Total'] == 115
        assert sampleSummary['Acquired'].loc['Study Sample', 'Total'] == 8
        assert sampleSummary['Acquired'].loc['Study Reference', 'Total'] == 11
        assert sampleSummary['Acquired'].loc['Long-Term Reference', 'Total'] == 1
        assert sampleSummary['Acquired'].loc['Serial Dilution', 'Total'] == 92
        assert sampleSummary['Acquired'].loc['Blank', 'Total'] == 2
        assert sampleSummary['Acquired'].loc['Unknown', 'Total'] == 1

        # Acquired - Marked for exclusion
        assert sampleSummary['Acquired'].loc['All', 'Marked for Exclusion'] == 1
        assert sampleSummary['Acquired'].loc['Study Sample', 'Marked for Exclusion'] == 1
        assert sampleSummary['Acquired'].loc['Study Reference', 'Marked for Exclusion'] == 0
        assert sampleSummary['Acquired'].loc['Long-Term Reference', 'Marked for Exclusion'] == 0
        assert sampleSummary['Acquired'].loc['Serial Dilution', 'Marked for Exclusion'] == 0
        assert sampleSummary['Acquired'].loc['Blank', 'Marked for Exclusion'] == 0
        assert sampleSummary['Acquired'].loc['Unknown', 'Marked for Exclusion'] == 0

        # Check details tables
        assert sampleSummary['MarkedToExclude Details'].shape == (1, 2)
        assert sampleSummary['UnknownType Details'].shape == (1, 1)

if __name__ == '__main__':
    unittest.main()
