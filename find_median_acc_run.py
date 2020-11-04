"""
Compute the median accuracy of our model and base line
"""
import statistics


if __name__ == "__main__":
    deepdrug_accs = [0.8035925313164737, # 40
                     0.8073741432285512,
                     0.8229732923658709,
                     0.8092649491845899,
                     0.8156464192862207,
                     0.8154100685417159,
                     0.812101158118648,
                     0.8163554715197353,
                     0.8135192625856772,
                     0.8187189789647837,
                     0.7993382179153864, # 50
                     0.8130465610966675,
                     0.8298274639565114,
                     0.8076104939730561,
                     0.8125738596076577,
                     0.8154100685417159,
                     0.813755613330182,
                     0.8123375088631529,
                     0.8158827700307256,
                     0.8095012999290948,
                     0.8090285984400851, # 60
                     0.8170645237532498,
                     0.7931930985582605,
                     0.8128102103521626,
                     0.8050106357835027]
    
    deepdrug_median = statistics.median(deepdrug_accs) # run 63
    print('median accuracy of classifier: ', deepdrug_median)

    gin_accs = [0.7307965020089813, # 100
                0.7612857480501064, 
                0.7508863152918932,
                0.7454502481682818,
                0.7546679272039707,
                0.7518317182699126,
                0.7523044197589223,
                0.7560860316709997,
                0.7478137556133302,
                0.7563223824155046,
                0.7456865989127865, # 110
                0.7468683526353108,
                0.7612857480501064,
                0.7430867407232333,
                0.7577404868825337,
                0.7485228078468447,
                0.7489955093358545,
                0.7515953675254077,
                0.7482864571023399,
                0.7601039943275821,
                0.7508863152918932, # 120
                0.7575041361380288,
                0.74048688253368,
                0.7551406286929804,
                0.7506499645473883] 

    gin_median = statistics.median(gin_accs) # run 102
    print('median accuracy of classifier: ', gin_median)
    