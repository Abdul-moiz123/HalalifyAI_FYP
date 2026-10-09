import unittest
import pandas as pd
import numpy as np
from create_targets import future_volatility, classify, FEATURES, TARGETS

class TargetChecks(unittest.TestCase):
    def test_uses_tomorrow_through_twenty_sessions(self):
        r=pd.Series(np.linspace(-0.03,0.06,45))
        actual=future_volatility(r)
        for i in [0,10,24]:
            self.assertAlmostEqual(actual.iloc[i], np.std(r.iloc[i+1:i+21],ddof=1)*np.sqrt(252),places=12)
        changed=r.copy(); changed.iloc[0]=100
        self.assertAlmostEqual(future_volatility(changed).iloc[0],actual.iloc[0],places=8)
        self.assertTrue(actual.iloc[-20:].isna().all())

    def test_missing_future_return_prevents_label(self):
        r=pd.Series(np.linspace(-0.02,0.04,45)); r.iloc[8]=np.nan
        labels=future_volatility(r)
        self.assertTrue(labels.iloc[:8].isna().all())
        self.assertTrue(pd.notna(labels.iloc[8]))

    def test_class_boundaries_and_missing(self):
        values=pd.Series([0.1,0.2,0.3,0.4,np.nan])
        result=classify(values,0.2,0.3)
        self.assertEqual(result.iloc[:4].tolist(),['low','low','medium','high'])
        self.assertTrue(pd.isna(result.iloc[4]))
        self.assertFalse(set(TARGETS).intersection(FEATURES))

if __name__=='__main__': unittest.main()
