

class hmmTagger:

    def getQ(self, paramList, hyperParam = None):
        """ compute q(t_n|t_1,t_2,...t_n-1) """

        if (not hyperParam):
            hyperParam = 1/(self.nOrder+1)*self.nOrder,

        if sum(hyperParam) != 1:
            raise HmmModel.INVALID_INTERPOLATION()

        c = self.tagsTransitions.getCount((t3,))
        bc = self.tagsTransitions.getCount((t2, t3))
        abc = self.tagsTransitions.getCount((t1, t2, t3))
        ab = self.tagsTransitions.getCount((t1, t2)) or 1
        b = self.tagsTransitions.getCount((t2,)) or 1
        tot = self.tagsTransitions.getCount() or 1
        # print("c:", c, "bc:", bc, "abc:", abc, "ab:", ab, "b:", b, "tot:", tot)

        return sum(np.array((abc / ab, bc / b, c / tot)) * hyperParam)