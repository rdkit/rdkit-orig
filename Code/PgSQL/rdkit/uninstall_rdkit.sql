
-- Adjust this setting to control where the objects get dropped.
SET search_path = public;


DROP OPERATOR CLASS IF EXISTS mol_ops USING hash CASCADE;
DROP OPERATOR CLASS IF EXISTS mol_ops USING gist CASCADE;
DROP OPERATOR CLASS IF EXISTS mol_ops USING btree CASCADE;
DROP OPERATOR CLASS IF EXISTS bfp_ops USING hash CASCADE;
DROP OPERATOR CLASS IF EXISTS bfp_ops USING gist CASCADE;
DROP OPERATOR CLASS IF EXISTS bfp_ops USING btree CASCADE;
DROP OPERATOR CLASS IF EXISTS sfp_ops USING hash CASCADE;
DROP OPERATOR CLASS IF EXISTS sfp_ops USING gist CASCADE;
DROP OPERATOR CLASS IF EXISTS sfp_ops USING btree CASCADE;
DROP OPERATOR CLASS IF EXISTS sfp_low_ops USING gist CASCADE;

DROP OPERATOR IF EXISTS <@ (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS @> (mol, mol) CASCADE;

DROP OPERATOR IF EXISTS % (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS # (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS % (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS # (sfp, sfp) CASCADE;

DROP FUNCTION IF EXISTS substruct(mol, mol) CASCADE;
DROP FUNCTION IF EXISTS rsubstruct(mol, mol) CASCADE;

DROP FUNCTION IF EXISTS tanimoto_sml(bfp, bfp) CASCADE;
DROP FUNCTION IF EXISTS dice_sml(bfp, bfp) CASCADE;
DROP FUNCTION IF EXISTS tanimoto_sml_op(bfp, bfp) CASCADE;
DROP FUNCTION IF EXISTS dice_sml_op(bfp, bfp) CASCADE;
DROP FUNCTION IF EXISTS tanimoto_sml(sfp, sfp) CASCADE;
DROP FUNCTION IF EXISTS dice_sml(sfp, sfp) CASCADE;
DROP FUNCTION IF EXISTS tanimoto_sml_op(sfp, sfp) CASCADE;
DROP FUNCTION IF EXISTS dice_sml_op(sfp, sfp) CASCADE;

DROP FUNCTION IF EXISTS size(bfp) CASCADE;

DROP OPERATOR IF EXISTS < (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS <= (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS >= (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS > (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS = (mol, mol) CASCADE;
DROP OPERATOR IF EXISTS <> (mol, mol) CASCADE;

DROP OPERATOR IF EXISTS < (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS <= (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS >= (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS > (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS = (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS <> (bfp, bfp) CASCADE;
DROP OPERATOR IF EXISTS < (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS <= (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS >= (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS > (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS = (sfp, sfp) CASCADE;
DROP OPERATOR IF EXISTS <> (sfp, sfp) CASCADE;

DROP FUNCTION IF EXISTS mol_cmp(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_lt(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_le(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_eq(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_ge(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_gt(mol,mol) CASCADE;
DROP FUNCTION IF EXISTS mol_ne(mol,mol) CASCADE;

DROP FUNCTION IF EXISTS bfp_cmp(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_lt(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_le(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_eq(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_ge(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_gt(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS bfp_ne(bfp,bfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_cmp(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_lt(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_le(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_eq(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_ge(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_gt(sfp,sfp) CASCADE;
DROP FUNCTION IF EXISTS sfp_ne(sfp,sfp) CASCADE;

DROP FUNCTION IF EXISTS gbfp_consistent(bytea,internal,int4) CASCADE;
DROP FUNCTION IF EXISTS gbfp_compress(internal) CASCADE;
DROP FUNCTION IF EXISTS gsfp_consistent(bytea,internal,int4) CASCADE;
DROP FUNCTION IF EXISTS gsfp_compress(internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_consistent(bytea,internal,int4) CASCADE;
DROP FUNCTION IF EXISTS gmol_compress(internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_decompress(internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_penalty(internal,internal,internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_picksplit(internal, internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_union(bytea, internal) CASCADE;
DROP FUNCTION IF EXISTS gmol_same(bytea, bytea, internal) CASCADE;

DROP FUNCTION IF EXISTS gslfp_consistent(bytea,internal,int4) CASCADE;
DROP FUNCTION IF EXISTS gslfp_compress(internal) CASCADE;
DROP FUNCTION IF EXISTS gslfp_decompress(internal) CASCADE;
DROP FUNCTION IF EXISTS gslfp_penalty(internal,internal,internal) CASCADE;
DROP FUNCTION IF EXISTS gslfp_picksplit(internal, internal) CASCADE;
DROP FUNCTION IF EXISTS gslfp_union(bytea, internal) CASCADE;
DROP FUNCTION IF EXISTS gslfp_same(bytea, bytea, internal) CASCADE;

DROP TYPE IF EXISTS mol CASCADE;
DROP TYPE IF EXISTS bfp CASCADE;
DROP TYPE IF EXISTS sfp CASCADE;
DROP TYPE IF EXISTS qmol CASCADE;

DROP FUNCTION IF EXISTS is_valid_smiles(cstring) CASCADE;
DROP FUNCTION IF EXISTS is_valid_smarts(cstring) CASCADE;
DROP FUNCTION IF EXISTS is_valid_ctab(cstring) CASCADE;
DROP FUNCTION IF EXISTS rdkit_version() CASCADE;
