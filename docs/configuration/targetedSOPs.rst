Generation of Targeted SOPs
---------------------------

To create a new pre-defined TargetLynx SOP (``fileType == 'TargetLynx'``) the `JSON` SOP must contain the following fields, the list must cover all compounds in the same order:

* ``methodName``
    The name of the method.

* ``chromatography``
    The chromatography employed.

* ``ionisation``
    The polarity employed.

* ``compoundID``
    A list of numeric ID (`"1","2",...`) that matches the TargetLynx compound ID.

* ``compoundName``
    A list of compound names.

* ``IS``
    A list of `"True" "False"` to indicate if the compound is an Internal Standard.

* ``unitFinal``
    A list of the compound measurement unit after application of the `unitCorrectionFactor` (can be left blank `""`).

* ``unitCorrectionFactor``
    A list of values by which to multiply the measured concentration in order to reach the `unitFinal` `("1","0.1","1000")`.

* ``calibrationMethod``
    A list of the calibration method employed for the compound, from enum CalibrationMethod: ``"noIS"`` `for compounds without Internal Standard (and Internal Standards themselves) = (use area)`, ``"backcalculatedIS"`` `for compounds using an Internal Standard = (use response)`, ``"noCalibration"`` `for compounds not quantified (Monitored for relative information)`.

* ``calibrationEquation``
    A list of equations to obtain the concentration given :math:`area`, :math:`responseFactor`, :math:`a` and :math:`b`. `responseFactor = (IS conc/IS Area)=response/Area (for noIS, responseFactor will be 1)` is automatically estimated from calibration samples.

   The calibration equation is only employed if values <LLOQ are replaced by the noise level (`TargetedDataset._targetLynxApplyLimitsOfQuantificationNoiseFilled()`)

    Calibration curve in TargetLynx is defined/established as: ``response = a * concentration + b (eq. 1)``
        * response is defined as: ``response = Area * (IS conc / IS Area) (eq. 2)`` [for 'noIS' response = Area]
        * using eq. 2, we can approximate the ratio IS Conc/IS Area in a representative sample as: ``responseFactor = response / area (eq. 3)``
        * Therefore: ``concentration = ((area*responseFactor) - b) / a (eq. 4)``
        * If in TargetLynx `'axis transformation'` is set to `log` (but still use `'Polynomial Type'=linear` and `'Fit Weighting'=None`)
        * eq.1 is changed to: ``log(response) = a * log(concentration) + b (eq. 5)``
        * and eq. 4 changed to: ``concentration = 10^( (log(area*responseFactor) - b) / a ) (eq. 5)``
        * Examples: ``"((area * responseFactor)-b)/a"``, ``"10**((numpy.log10(area * responseFactor)-b)/a)"``, ``"area/a"`` | if b not needed, set to 0 in csv [use for linear noIS, area=response, responseFactor=1, and response = a * concentration ]

* ``quantificationType``
    A list of the type of quantification employed, from enum QuantificationType: ``"IS"``, ``"QuantOwnLabeledAnalogue"``, ``"QuantAltLabeledAnalogue"``, ``"QuantOther"`` or ``"Monitored"``

    .. Note:: `quantificationType` ``"IS"`` must match with `IS` ``"True"``. `quantificationType` ``"Monitored"`` must match with `calibrationMethod` ``"noCalibration"``.

* ``externalID``
   A list of external ID, each external ID must also be present as its own field as a list of identifier (for that external ID). For example, if ``"externalID":["PubChem ID"]``, the field ``"PubChem ID":["ID1","ID2","","ID75"]"`` is required.

* ``sampleMetadataNotExported``
   A list of ``sampleMetadata`` columns to exclude from export and reports.

* ``featureMetadataNotExported``
   A list of ``featureMetadata`` columns to exclude from export and reports.

.. literalinclude:: ../../nPYc/StudyDesigns/SOP/AminoAcidMS.json
   :caption: Example TargetLynx SOP for the Amino Acids assay (Gray N. `et al`. Human Plasma and Serum via Precolumn Derivatization with 6‑Aminoquinolyl‑N‑hydroxysuccinimidyl Carbamate: Application to Acetaminophen-Induced Liver Failure. `Analytical Chemistry`, 89, 2017, 2478−2487)
