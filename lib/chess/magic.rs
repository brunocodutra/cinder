use crate::chess::{Bitboard, Square};
use crate::util::Int;

#[derive(Debug, Copy, Hash)]
#[derive_const(Clone, Eq, PartialEq)]
pub struct Magic(Bitboard, u64, usize);

impl Magic {
    #[inline(always)]
    pub const fn mask(&self) -> Bitboard {
        self.0
    }

    #[inline(always)]
    pub const fn factor(&self) -> u64 {
        self.1
    }

    #[inline(always)]
    pub const fn offset(&self) -> usize {
        self.2
    }

    /// Bishop fixed shift magics by Volker Annuss,
    /// see <http://www.talkchess.com/forum/viewtopic.php?p=727500&t=64790>.
    #[inline(always)]
    pub const fn bishop(sq: Square) -> Self {
        const MAGICS: [Magic; Square::MAX as usize + 1] = [
            Magic(Bitboard::new(0x0040201008040200), 0x007FBFBFBFBFBFFF, 5378),
            Magic(Bitboard::new(0x0000402010080400), 0x0000A060401007FC, 4093),
            Magic(Bitboard::new(0x0000004020100A00), 0x0001004008020000, 4314),
            Magic(Bitboard::new(0x0000000040221400), 0x0000806004000000, 6587),
            Magic(Bitboard::new(0x0000000002442800), 0x0000100400000000, 6491),
            Magic(Bitboard::new(0x0000000204085000), 0x000021C100B20000, 6330),
            Magic(Bitboard::new(0x0000020408102000), 0x0000040041008000, 5609),
            Magic(Bitboard::new(0x0002040810204000), 0x00000FB0203FFF80, 22236),
            Magic(Bitboard::new(0x0020100804020000), 0x0000040100401004, 6106),
            Magic(Bitboard::new(0x0040201008040000), 0x0000020080200802, 5625),
            Magic(Bitboard::new(0x00004020100A0000), 0x0000004010202000, 16785),
            Magic(Bitboard::new(0x0000004022140000), 0x0000008060040000, 16817),
            Magic(Bitboard::new(0x0000000244280000), 0x0000004402000000, 6842),
            Magic(Bitboard::new(0x0000020408500000), 0x0000000801008000, 7003),
            Magic(Bitboard::new(0x0002040810200000), 0x000007EFE0BFFF80, 4197),
            Magic(Bitboard::new(0x0004081020400000), 0x0000000820820020, 7356),
            Magic(Bitboard::new(0x0010080402000200), 0x0000400080808080, 4602),
            Magic(Bitboard::new(0x0020100804000400), 0x00021F0100400808, 4538),
            Magic(Bitboard::new(0x004020100A000A00), 0x00018000C06F3FFF, 29531),
            Magic(Bitboard::new(0x0000402214001400), 0x0000258200801000, 45393),
            Magic(Bitboard::new(0x0000024428002800), 0x0000240080840000, 12420),
            Magic(Bitboard::new(0x0002040850005000), 0x000018000C03FFF8, 15763),
            Magic(Bitboard::new(0x0004081020002000), 0x00000A5840208020, 5050),
            Magic(Bitboard::new(0x0008102040004000), 0x0000020008208020, 4346),
            Magic(Bitboard::new(0x0008040200020400), 0x0000804000810100, 6074),
            Magic(Bitboard::new(0x0010080400040800), 0x0001011900802008, 7866),
            Magic(Bitboard::new(0x0020100A000A1000), 0x0000804000810100, 32139),
            Magic(Bitboard::new(0x0040221400142200), 0x000100403C0403FF, 57673),
            Magic(Bitboard::new(0x0002442800284400), 0x00078402A8802000, 55365),
            Magic(Bitboard::new(0x0004085000500800), 0x0000101000804400, 15818),
            Magic(Bitboard::new(0x0008102000201000), 0x0000080800104100, 5562),
            Magic(Bitboard::new(0x0010204000402000), 0x00004004C0082008, 6390),
            Magic(Bitboard::new(0x0004020002040800), 0x0001010120008020, 7930),
            Magic(Bitboard::new(0x0008040004081000), 0x000080809A004010, 13329),
            Magic(Bitboard::new(0x00100A000A102000), 0x0007FEFE08810010, 7170),
            Magic(Bitboard::new(0x0022140014224000), 0x0003FF0F833FC080, 27267),
            Magic(Bitboard::new(0x0044280028440200), 0x007FE08019003042, 53787),
            Magic(Bitboard::new(0x0008500050080400), 0x003FFFEFEA003000, 5097),
            Magic(Bitboard::new(0x0010200020100800), 0x0000101010002080, 6643),
            Magic(Bitboard::new(0x0020400040201000), 0x0000802005080804, 6138),
            Magic(Bitboard::new(0x0002000204081000), 0x0000808080A80040, 7418),
            Magic(Bitboard::new(0x0004000408102000), 0x0000104100200040, 7898),
            Magic(Bitboard::new(0x000A000A10204000), 0x0003FFDF7F833FC0, 42012),
            Magic(Bitboard::new(0x0014001422400000), 0x0000008840450020, 57350),
            Magic(Bitboard::new(0x0028002844020000), 0x00007FFC80180030, 22813),
            Magic(Bitboard::new(0x0050005008040200), 0x007FFFDD80140028, 56693),
            Magic(Bitboard::new(0x0020002010080400), 0x00020080200A0004, 5818),
            Magic(Bitboard::new(0x0040004020100800), 0x0000101010100020, 7098),
            Magic(Bitboard::new(0x0000020408102000), 0x0007FFDFC1805000, 4451),
            Magic(Bitboard::new(0x0000040810204000), 0x0003FFEFE0C02200, 4709),
            Magic(Bitboard::new(0x00000A1020400000), 0x0000000820806000, 4794),
            Magic(Bitboard::new(0x0000142240000000), 0x0000000008403000, 13364),
            Magic(Bitboard::new(0x0000284402000000), 0x0000000100202000, 4570),
            Magic(Bitboard::new(0x0000500804020000), 0x0000004040802000, 4282),
            Magic(Bitboard::new(0x0000201008040200), 0x0004010040100400, 14964),
            Magic(Bitboard::new(0x0000402010080400), 0x00006020601803F4, 4026),
            Magic(Bitboard::new(0x0002040810204000), 0x0003FFDFDFC28048, 4826),
            Magic(Bitboard::new(0x0004081020400000), 0x0000000820820020, 7354),
            Magic(Bitboard::new(0x000A102040000000), 0x0000000008208060, 4848),
            Magic(Bitboard::new(0x0014224000000000), 0x0000000000808020, 15946),
            Magic(Bitboard::new(0x0028440200000000), 0x0000000001002020, 14932),
            Magic(Bitboard::new(0x0050080402000000), 0x0000000401002008, 16588),
            Magic(Bitboard::new(0x0020100804020000), 0x0000004040404040, 6905),
            Magic(Bitboard::new(0x0040201008040200), 0x007FFF9FDF7FF813, 16076),
        ];

        MAGICS[sq as usize]
    }

    /// Rook fixed shift magics by Volker Annuss,
    /// see <http://www.talkchess.com/forum/viewtopic.php?p=727500&t=64790>.
    #[inline(always)]
    pub const fn rook(sq: Square) -> Self {
        const MAGICS: [Magic; Square::MAX as usize + 1] = [
            Magic(Bitboard::new(0x000101010101017E), 0x00280077FFEBFFFE, 26304),
            Magic(Bitboard::new(0x000202020202027C), 0x2004010201097FFF, 35520),
            Magic(Bitboard::new(0x000404040404047A), 0x0010020010053FFF, 38592),
            Magic(Bitboard::new(0x0008080808080876), 0x0040040008004002, 8026),
            Magic(Bitboard::new(0x001010101010106E), 0x7FD00441FFFFD003, 22196),
            Magic(Bitboard::new(0x002020202020205E), 0x4020008887DFFFFE, 80870),
            Magic(Bitboard::new(0x004040404040403E), 0x004000888847FFFF, 76747),
            Magic(Bitboard::new(0x008080808080807E), 0x006800FBFF75FFFD, 30400),
            Magic(Bitboard::new(0x0001010101017E00), 0x000028010113FFFF, 11115),
            Magic(Bitboard::new(0x0002020202027C00), 0x0020040201FCFFFF, 18205),
            Magic(Bitboard::new(0x0004040404047A00), 0x007FE80042FFFFE8, 53577),
            Magic(Bitboard::new(0x0008080808087600), 0x00001800217FFFE8, 62724),
            Magic(Bitboard::new(0x0010101010106E00), 0x00001800073FFFE8, 34282),
            Magic(Bitboard::new(0x0020202020205E00), 0x00001800E05FFFE8, 29196),
            Magic(Bitboard::new(0x0040404040403E00), 0x00001800602FFFE8, 23806),
            Magic(Bitboard::new(0x0080808080807E00), 0x000030002FFFFFA0, 49481),
            Magic(Bitboard::new(0x00010101017E0100), 0x00300018010BFFFF, 2410),
            Magic(Bitboard::new(0x00020202027C0200), 0x0003000C0085FFFB, 36498),
            Magic(Bitboard::new(0x00040404047A0400), 0x0004000802010008, 24478),
            Magic(Bitboard::new(0x0008080808760800), 0x0004002020020004, 10074),
            Magic(Bitboard::new(0x00101010106E1000), 0x0001002002002001, 79315),
            Magic(Bitboard::new(0x00202020205E2000), 0x0001001000801040, 51779),
            Magic(Bitboard::new(0x00404040403E4000), 0x0000004040008001, 13586),
            Magic(Bitboard::new(0x00808080807E8000), 0x0000006800CDFFF4, 19323),
            Magic(Bitboard::new(0x000101017E010100), 0x0040200010080010, 70612),
            Magic(Bitboard::new(0x000202027C020200), 0x0000080010040010, 83652),
            Magic(Bitboard::new(0x000404047A040400), 0x0004010008020008, 63110),
            Magic(Bitboard::new(0x0008080876080800), 0x0000040020200200, 34496),
            Magic(Bitboard::new(0x001010106E101000), 0x0002008010100100, 84966),
            Magic(Bitboard::new(0x002020205E202000), 0x0000008020010020, 54341),
            Magic(Bitboard::new(0x004040403E404000), 0x0000008020200040, 60421),
            Magic(Bitboard::new(0x008080807E808000), 0x0000820020004020, 86402),
            Magic(Bitboard::new(0x0001017E01010100), 0x00FFFD1800300030, 50245),
            Magic(Bitboard::new(0x0002027C02020200), 0x007FFF7FBFD40020, 76622),
            Magic(Bitboard::new(0x0004047A04040400), 0x003FFFBD00180018, 84676),
            Magic(Bitboard::new(0x0008087608080800), 0x001FFFDE80180018, 78757),
            Magic(Bitboard::new(0x0010106E10101000), 0x000FFFE0BFE80018, 37346),
            Magic(Bitboard::new(0x0020205E20202000), 0x0001000080202001, 370),
            Magic(Bitboard::new(0x0040403E40404000), 0x0003FFFBFF980180, 42182),
            Magic(Bitboard::new(0x0080807E80808000), 0x0001FFFDFF9000E0, 45385),
            Magic(Bitboard::new(0x00017E0101010100), 0x00FFFEFEEBFFD800, 61659),
            Magic(Bitboard::new(0x00027C0202020200), 0x007FFFF7FFC01400, 12790),
            Magic(Bitboard::new(0x00047A0404040400), 0x003FFFBFE4FFE800, 16762),
            Magic(Bitboard::new(0x0008760808080800), 0x001FFFF01FC03000, 0),
            Magic(Bitboard::new(0x00106E1010101000), 0x000FFFE7F8BFE800, 38380),
            Magic(Bitboard::new(0x00205E2020202000), 0x0007FFDFDF3FF808, 11098),
            Magic(Bitboard::new(0x00403E4040404000), 0x0003FFF85FFFA804, 21803),
            Magic(Bitboard::new(0x00807E8080808000), 0x0001FFFD75FFA802, 39189),
            Magic(Bitboard::new(0x007E010101010100), 0x00FFFFD7FFEBFFD8, 58628),
            Magic(Bitboard::new(0x007C020202020200), 0x007FFF75FF7FBFD8, 44116),
            Magic(Bitboard::new(0x007A040404040400), 0x003FFF863FBF7FD8, 78357),
            Magic(Bitboard::new(0x0076080808080800), 0x001FFFBFDFD7FFD8, 44481),
            Magic(Bitboard::new(0x006E101010101000), 0x000FFFF810280028, 64134),
            Magic(Bitboard::new(0x005E202020202000), 0x0007FFD7F7FEFFD8, 41759),
            Magic(Bitboard::new(0x003E404040404000), 0x0003FFFC0C480048, 1394),
            Magic(Bitboard::new(0x007E808080808000), 0x0001FFFFAFD7FFD8, 40910),
            Magic(Bitboard::new(0x7E01010101010100), 0x00FFFFE4FFDFA3BA, 66516),
            Magic(Bitboard::new(0x7C02020202020200), 0x007FFFEF7FF3D3DA, 3897),
            Magic(Bitboard::new(0x7A04040404040400), 0x003FFFBFDFEFF7FA, 3930),
            Magic(Bitboard::new(0x7608080808080800), 0x001FFFEFF7FBFC22, 72934),
            Magic(Bitboard::new(0x6E10101010101000), 0x0000020408001001, 72662),
            Magic(Bitboard::new(0x5E20202020202000), 0x0007FFFEFFFF77FD, 56325),
            Magic(Bitboard::new(0x3E40404040404000), 0x0003FFFFBF7DFEEC, 66501),
            Magic(Bitboard::new(0x7E80808080808000), 0x0001FFFF9DFFA333, 14826),
        ];

        MAGICS[sq as usize]
    }
}
